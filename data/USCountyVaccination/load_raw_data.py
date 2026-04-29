"""Download/cache raw inputs for the USCountyVaccination workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    BANSAL_VACCINATION_URL,
    CDC_VACCINATION_URL,
    TIGER_2021_COUNTY_URL,
    TIGER_2022_COUNTY_URL,
    ensure_directories,
)
from data_utils import download_if_missing  # noqa: E402


NYT_COUNTY_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/cache raw US county COVID, vaccination, and geography inputs."
    )
    parser.add_argument(
        "--include_tiger_2022_fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also cache the 2022 TIGER county shapefile fallback.",
    )
    return parser.parse_args()


def load_raw_data(include_tiger_2022_fallback: bool = True) -> None:
    paths = ensure_directories()
    downloads = [
        (NYT_COUNTY_URL, paths["raw_nyt"] / "us-counties.csv"),
        (CDC_VACCINATION_URL, paths["raw_vaccination"] / "cdc_county_vaccinations.csv"),
        (BANSAL_VACCINATION_URL, paths["raw_vaccination"] / "bansal_data_county_timeseries.csv"),
        (TIGER_2021_COUNTY_URL, paths["raw_geography"] / "tl_2021_us_county.zip"),
    ]
    if include_tiger_2022_fallback:
        downloads.append(
            (TIGER_2022_COUNTY_URL, paths["raw_geography"] / "tl_2022_us_county.zip")
        )
    for url, target in downloads:
        download_if_missing(url, target)


def main() -> None:
    args = parse_args()
    load_raw_data(include_tiger_2022_fallback=bool(args.include_tiger_2022_fallback))


if __name__ == "__main__":
    main()
