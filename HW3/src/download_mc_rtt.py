"""Download MC_RTT directly from DANDI, with no DANDI client dependency."""
from urllib.request import urlretrieve

from .data import DATA


def main():
    destination = DATA / "mc_rtt"
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "sub-Indy_desc-train_behavior+ecephys.nwb"
    url = (
        "https://api.dandiarchive.org/api/dandisets/000129/versions/draft/"
        "assets/2ae6bf3c-788b-4ece-8c01-4b4a5680b25b/download/"
    )
    if output.exists() and output.stat().st_size > 40_000_000:
        print(f"Already downloaded: {output}")
        return
    print(f"Downloading MC_RTT to {output}")
    urlretrieve(url, output)


if __name__ == "__main__":
    main()
