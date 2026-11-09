"""Small allowlist of well-known registered domains, used by the app only (not by the model).

Why: every legitimate PhiUSIIL URL is a bare ``https://www.<domain>`` homepage, so the
model has seen almost no legitimate subdomains (accounts.google.com, en.wikipedia.org).
It therefore over-flags subdomains of large sites. Real filters combine models with
reputation lists, and this list plays that role. It is applied *after* the model, it is
shown in the UI, and it never applies to user-content platforms (see FREE_HOSTING),
because sites.google.com, *.github.io etc. regularly host phishing pages.
"""

KNOWN_DOMAINS = frozenset({
    "google.com", "google.co.uk", "google.co.in", "google.de", "youtube.com", "gmail.com",
    "wikipedia.org", "wikimedia.org", "github.com", "gitlab.com", "stackoverflow.com",
    "microsoft.com", "live.com", "office.com", "microsoftonline.com", "outlook.com",
    "bing.com", "apple.com", "icloud.com", "amazon.com", "amazon.co.uk", "amazon.in",
    "amazon.de", "aws.amazon.com", "facebook.com", "instagram.com", "whatsapp.com",
    "linkedin.com", "twitter.com", "x.com", "reddit.com", "netflix.com", "spotify.com",
    "paypal.com", "ebay.com", "yahoo.com", "yahoo.co.jp", "baidu.com", "zoom.us",
    "dropbox.com", "adobe.com", "salesforce.com", "slack.com", "discord.com",
    "openai.com", "anthropic.com", "claude.ai", "chatgpt.com", "cloudflare.com",
    "mozilla.org", "python.org", "pypi.org", "npmjs.com", "medium.com", "bbc.co.uk",
    "bbc.com", "cnn.com", "nytimes.com", "theguardian.com", "chase.com",
    "bankofamerica.com", "wellsfargo.com", "hsbc.com", "irs.gov", "gov.uk",
    "nasa.gov", "who.int", "un.org", "mit.edu", "stanford.edu", "harvard.edu",
    "coursera.org", "udemy.com", "kaggle.com", "uci.edu", "booking.com",
    "airbnb.com", "uber.com", "walmart.com", "target.com", "etsy.com", "twitch.tv",
    "tiktok.com", "pinterest.com", "quora.com", "imdb.com", "duckduckgo.com",
})
