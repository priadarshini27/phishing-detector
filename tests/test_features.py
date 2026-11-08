import math

import pytest

from src.features import (
    FEATURE_COLUMNS, extract_features, extract_host_features, normalize_url, registered_domain,
    url_warnings,
)

VALID_URLS = [
    "https://www.google.com",
    "example.com/login",
    "  HTTPS://WWW.Example.COM/Path  ",
    "http://192.168.0.1:8080/admin",
    "https://www.bbc.co.uk/news",
    "https://a.b.c.d.example.com",
    "http://paypal.com@evil.xyz/verify",
    "https://xn--mnchen-3ya.de",
    "https://münchen.de",
    "https://sub-domain-with-hyphens.example.net",
    "https://example.com/" + "a" * 1500,
    "https://example.com?q=%20test",
    "https://x.firebaseapp.com",
    "http://[2001:db8::1]/path",
    "https://shop.example.com.au/item?id=5&x=2",
    "www.example.org",
    "//example.io/x",
    "https://EXAMPLE.com:443/",
    "https://1234567.top",
    "https://login-secure-account.web.app",
]


@pytest.mark.parametrize("url", VALID_URLS)
def test_all_features_present_and_numeric(url):
    f = extract_features(url)
    assert list(f) == FEATURE_COLUMNS
    for k, v in f.items():
        if k == "tld_group":
            assert isinstance(v, str) and v
        else:
            assert isinstance(v, (int, float)) and math.isfinite(v), k


def test_normalisation():
    p = normalize_url("  HTTPS://WWW.Example.COM:8443/Path?q=1  ")
    assert (p.scheme, p.host, p.port, p.path, p.query) == ("https", "example.com", 8443, "/Path", "q=1")
    assert normalize_url("example.com/login").scheme == ""  # no scheme typed
    assert normalize_url("example.com/login").host == "example.com"


def test_same_host_same_features():
    a = extract_features("https://www.example.com")
    b = extract_features("http://example.com/some/deep/path?x=1")
    assert a == b  # the model only sees the hostname


@pytest.mark.parametrize("bad", ["", "   ", None, "ftp://example.com", "localhost",
                                 "http://exa mple.com", "https://" + "a" * 3000 + ".com",
                                 "http://example.com:99999"])
def test_invalid_input(bad):
    with pytest.raises(ValueError):
        normalize_url(bad)


def test_ip_and_userinfo():
    p = normalize_url("http://paypal.com@10.0.0.1/x")
    assert p.host == "10.0.0.1" and p.has_userinfo
    f = extract_host_features(p.host)
    assert f["is_ip"] == 1 and f["tld_group"] == "ip"
    w = " ".join(url_warnings(p))
    assert "@" in w and "IP" in w


def test_registered_domain_and_suffix():
    assert registered_domain("news.bbc.co.uk") == "bbc.co.uk"
    assert registered_domain("abc.firebaseapp.com") == "firebaseapp.com"
    f = extract_host_features("news.bbc.co.uk")
    assert f["num_subdomains"] == 1 and f["is_multi_part_suffix"] == 1 and f["tld_group"] == "uk"


def test_flags():
    f = extract_host_features("emerald-exch-0utlook.firebaseapp.com")
    assert f["is_free_hosting"] == 1 and f["brand_in_subdomain"] == 1
    assert extract_host_features("paypal.com")["brand_in_subdomain"] == 0
    assert extract_host_features("groups.example.com")["brand_in_subdomain"] == 0  # "ups"
    assert extract_host_features("secure-login.example.com")["num_suspicious_keywords"] >= 2
    assert extract_host_features("xn--pypal-4ve.com")["is_punycode"] == 1


def test_warnings_clean_url():
    assert url_warnings(normalize_url("https://github.com/user/repo")) == []
