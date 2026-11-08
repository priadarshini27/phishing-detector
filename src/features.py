"""Shared URL feature extractor: the single source of truth for training and the app.

The model only sees HOSTNAME features. In PhiUSIIL every legitimate URL has the form
``https://www.<domain>`` (no path, no query, always HTTPS), so any path, query or
scheme feature would teach the model a dataset artefact rather than real phishing
signals. Path, query and scheme are still inspected by ``url_warnings`` as
rule-based red flags, which are shown in the app but never fed to the model.

Nothing here uses dataset-wide statistics or labels; every value is computed from
the single URL string passed in.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlsplit

import tldextract

# Offline extractor that uses the public-suffix snapshot bundled with tldextract.
# Private suffixes (e.g. firebaseapp.com) are NOT treated as suffixes, so
# "x.firebaseapp.com" has registered domain "firebaseapp.com".
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

MAX_URL_LENGTH = 2048

# Frozen: the 30 most frequent last-label TLDs in PhiUSIIL. Anything else -> "other".
TOP_TLDS = (
    "com", "org", "net", "app", "uk", "co", "io", "de", "ru", "au",
    "dev", "top", "jp", "it", "edu", "fr", "br", "nl", "ca", "info",
    "site", "xyz", "link", "in", "pl", "gov", "cf", "ga", "us", "me",
)

# Well-known free hosting / site-builder / app platforms (domain knowledge, fixed list).
FREE_HOSTING = (
    "web.app", "firebaseapp.com", "repl.co", "replit.dev", "replit.app",
    "weebly.com", "weeblysite.com", "ipfs.io", "dweb.link", "workers.dev",
    "pages.dev", "square.site", "pantheonsite.io", "godaddysites.com",
    "github.io", "gitlab.io", "netlify.app", "vercel.app", "herokuapp.com",
    "blogspot.com", "wixsite.com", "wix.com", "000webhostapp.com", "glitch.me",
    "firebaseio.com", "appspot.com", "azurewebsites.net", "web.core.windows.net",
    "cloudfront.net", "ngrok.io", "ngrok-free.app", "onrender.com", "fly.dev",
    "surge.sh", "webflow.io", "wordpress.com", "sites.google.com", "carrd.co",
    "mystrikingly.com", "jimdosite.com", "yolasite.com", "ukit.me", "tilda.ws",
    "framer.app", "glide.page", "myshopify.com", "translate.goog", "my.id",
    "duckdns.org", "xsph.ru", "000webhost.com", "byethost.com", "freehosting.com",
)

SUSPICIOUS_KEYWORDS = (
    "login", "logon", "signin", "sign-in", "verify", "verification", "secure",
    "account", "update", "bank", "wallet", "support", "mail", "auth", "confirm",
    "billing", "payment", "recover", "unlock", "webscr", "service", "client",
    "customer", "security", "alert", "claim", "bonus", "free", "gift", "prize",
    "airdrop", "connect", "validate", "restore", "helpdesk", "official",
)

BRANDS = (
    "paypal", "apple", "icloud", "microsoft", "outlook", "office", "office365",
    "hotmail", "live", "google", "gmail", "amazon", "netflix", "facebook",
    "instagram", "whatsapp", "meta", "chase", "wellsfargo", "bankofamerica",
    "citi", "hsbc", "barclays", "santander", "dhl", "fedex", "ups", "usps",
    "ebay", "yahoo", "aol", "adobe", "dropbox", "docusign", "linkedin",
    "twitter", "steam", "coinbase", "binance", "metamask", "blockchain",
    "americanexpress", "amex", "visa", "mastercard", "att", "verizon",
    "spotify", "roblox", "discord", "telegram", "sharepoint", "onedrive",
)

_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a"})
_VOWELS = set("aeiou")
_HOST_RE = re.compile(r"^[a-z0-9.-]+$")

FEATURE_COLUMNS = [
    # size
    "host_length", "num_labels", "num_subdomains", "sld_length",
    "longest_label_len", "mean_label_len",
    # characters
    "num_hyphens", "num_digits", "digit_ratio", "vowel_ratio",
    "max_consonant_run", "max_digit_run", "letter_digit_switches",
    # randomness
    "host_entropy", "sld_entropy",
    # flags
    "is_ip", "is_punycode", "is_multi_part_suffix", "is_free_hosting",
    # words
    "num_suspicious_keywords", "brand_in_subdomain",
    # tld
    "tld_length", "tld_group",
]
CATEGORICAL = ["tld_group"]
NUMERIC = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL]


@dataclass(frozen=True)
class ParsedURL:
    raw: str
    scheme: str          # "" if the user typed no scheme
    host: str            # lowercase, no port, no leading "www."
    port: int | None
    path: str
    query: str
    has_userinfo: bool   # e.g. http://paypal.com@evil.com


def normalize_url(raw: str) -> ParsedURL:
    """Parse a user- or dataset-supplied URL. Raises ValueError for unusable input."""
    if raw is None:
        raise ValueError("Please enter a URL.")
    url = str(raw).strip()
    if not url:
        raise ValueError("Please enter a URL.")
    if len(url) > MAX_URL_LENGTH:
        raise ValueError(f"URL is longer than {MAX_URL_LENGTH} characters.")

    scheme_given = ""
    m = re.match(r"^([a-zA-Z][a-zA-Z0-9+.-]*)://", url)
    if m:
        scheme_given = m.group(1).lower()
        if scheme_given not in ("http", "https"):
            raise ValueError(f"Only http/https URLs are supported (got '{scheme_given}').")
        to_parse = url
    else:
        to_parse = "http://" + url.lstrip("/")

    try:
        parts = urlsplit(to_parse)
        host = parts.hostname or ""
        port = parts.port
    except ValueError as exc:  # bad port, bad IPv6 brackets ...
        raise ValueError(f"Could not parse URL: {exc}") from None

    host = host.strip().rstrip(".").lower()
    if not host:
        raise ValueError("Could not find a hostname in the URL.")
    if " " in host:
        raise ValueError("Hostname contains spaces.")
    if not _HOST_RE.match(host):
        # Internationalised hostnames -> punycode.
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError:
            raise ValueError("Hostname contains invalid characters.") from None
    if not _is_ip(host) and "." not in host:
        raise ValueError("Hostname needs a domain and a TLD, e.g. example.com.")
    if host.startswith("www."):
        host = host[4:]

    return ParsedURL(
        raw=url,
        scheme=scheme_given,
        host=host,
        port=port,
        path=parts.path or "",
        query=parts.query or "",
        has_userinfo="@" in parts.netloc,
    )


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def _max_run(s: str, pred) -> int:
    best = cur = 0
    for ch in s:
        cur = cur + 1 if pred(ch) else 0
        best = max(best, cur)
    return best


@lru_cache(maxsize=131072)
def split_host(host: str):
    """Return (subdomain, domain, suffix, registered_domain) for a normalised host."""
    if _is_ip(host):
        return "", host, "", host
    r = _EXTRACT(host)
    if r.suffix and r.domain:
        reg = f"{r.domain}.{r.suffix}"
    else:  # unknown suffix: fall back to the last two labels
        labels = host.split(".")
        reg = ".".join(labels[-2:])
    return r.subdomain, r.domain or host.split(".")[0], r.suffix, reg


def registered_domain(host: str) -> str:
    return split_host(host)[3]


def matched_free_host(host: str) -> str | None:
    for platform in FREE_HOSTING:
        if host == platform or host.endswith("." + platform):
            return platform
    return None


def found_keywords(text: str) -> list[str]:
    return [k for k in SUSPICIOUS_KEYWORDS if k in text]


def found_brand_in_subdomain(host: str) -> str | None:
    """Brand name in the host but not as the registered domain itself."""
    if _is_ip(host):
        return None
    sub, domain, _suffix, _reg = split_host(host)
    domain_plain = domain.translate(_LEET)
    rest = (sub + "." + domain).translate(_LEET) if sub else domain_plain
    for brand in BRANDS:
        if domain_plain == brand or domain == brand:
            continue  # the real brand domain (paypal.com) or a ccTLD variant
        if len(brand) <= 4:
            # short brands only as a whole token, to avoid "ups" in "groups"
            tokens = re.split(r"[.\-_]", rest)
            if brand in tokens:
                return brand
        elif brand in rest:
            return brand
    return None


def extract_host_features(host: str) -> dict:
    """Compute the model's features from a normalised hostname (see normalize_url)."""
    host = host.lower()
    is_ip = _is_ip(host)
    sub, domain, suffix, _reg = split_host(host)
    labels = [lab for lab in host.split(".") if lab]
    tld = "" if is_ip else (labels[-1] if labels else "")
    letters = sum(ch.isalpha() for ch in host)
    digits = sum(ch.isdigit() for ch in host)
    alnum = letters + digits

    switches = 0
    prev = None
    for ch in host:
        kind = "d" if ch.isdigit() else "l" if ch.isalpha() else None
        if kind and prev and kind != prev:
            switches += 1
        if kind:
            prev = kind

    return {
        "host_length": len(host),
        "num_labels": len(labels),
        "num_subdomains": 0 if is_ip else (len(sub.split(".")) if sub else 0),
        "sld_length": len(domain),
        "longest_label_len": max((len(lab) for lab in labels), default=0),
        "mean_label_len": round(sum(len(lab) for lab in labels) / max(len(labels), 1), 4),
        "num_hyphens": host.count("-"),
        "num_digits": digits,
        "digit_ratio": round(digits / max(len(host), 1), 4),
        "vowel_ratio": round(sum(ch in _VOWELS for ch in host) / max(letters, 1), 4),
        "max_consonant_run": _max_run(host, lambda c: c.isalpha() and c not in _VOWELS),
        "max_digit_run": _max_run(host, str.isdigit),
        "letter_digit_switches": switches if alnum else 0,
        "host_entropy": round(_entropy(host), 4),
        "sld_entropy": round(_entropy(domain), 4),
        "is_ip": int(is_ip),
        "is_punycode": int("xn--" in host),
        "is_multi_part_suffix": int("." in suffix),
        "is_free_hosting": int(matched_free_host(host) is not None),
        "num_suspicious_keywords": len(found_keywords(host)),
        "brand_in_subdomain": int(found_brand_in_subdomain(host) is not None),
        "tld_length": len(tld),
        "tld_group": tld if tld in TOP_TLDS else ("ip" if is_ip else "other"),
    }


def extract_features(url: str) -> dict:
    """Convenience: URL string -> feature dict (raises ValueError on bad input)."""
    return extract_host_features(normalize_url(url).host)


def url_warnings(p: ParsedURL) -> list[str]:
    """Rule-based red flags from the full URL. Shown in the app, NOT used by the model."""
    w = []
    if p.has_userinfo:
        w.append("Contains '@' before the host: browsers ignore everything before it, "
                 "a classic trick to disguise the real destination.")
    if _is_ip(p.host):
        w.append("Uses a raw IP address instead of a domain name.")
    if p.port and p.port not in (80, 443):
        w.append(f"Uses a non-standard port ({p.port}).")
    if p.scheme == "http":
        w.append("Uses plain HTTP (no TLS encryption).")
    if "%" in p.path or "%" in p.query or "%" in p.host:
        w.append("Contains percent-encoded characters, which can hide the real text.")
    if len(p.raw) > 100:
        w.append(f"Very long URL ({len(p.raw)} characters).")
    kw = found_keywords((p.path + "?" + p.query).lower())
    if kw:
        w.append("Path or query contains sensitive keywords: " + ", ".join(sorted(set(kw))[:5]) + ".")
    if "//" in p.path:
        w.append("Path contains '//', which is often used for redirects.")
    if p.host.count(".") >= 4:
        w.append(f"Many subdomain levels ({p.host.count('.') + 1} labels).")
    if "xn--" in p.host:
        w.append("Punycode hostname: it may imitate another domain with look-alike characters.")
    return w
