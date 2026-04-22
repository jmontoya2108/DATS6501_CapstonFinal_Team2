from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "data" / "Location Coordinates.xlsx"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "Private_coordinates.xlsx"

# The Streamlit app expects these tab labels. The source workbook used during
# development had origin/destination tab names reversed, so this mapping keeps
# the app-compatible workbook structure.
SHEET_MAPPING = {
    "Destinations": "Origins",
    "Origins": "Destinations",
}


def displace_point(lat, lon, min_km=10, max_km=20):
    """Randomly displace a geographic point between min_km and max_km."""
    earth_radius_km = 6371
    distance = np.random.uniform(min_km, max_km)
    bearing = np.random.uniform(0, 2 * np.pi)

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    new_lat = np.arcsin(
        np.sin(lat_rad) * np.cos(distance / earth_radius_km)
        + np.cos(lat_rad) * np.sin(distance / earth_radius_km) * np.cos(bearing)
    )

    new_lon = lon_rad + np.arctan2(
        np.sin(bearing) * np.sin(distance / earth_radius_km) * np.cos(lat_rad),
        np.cos(distance / earth_radius_km) - np.sin(lat_rad) * np.sin(new_lat),
    )

    return np.degrees(new_lat), np.degrees(new_lon)


def create_private_coordinates(input_file=DEFAULT_INPUT_FILE, output_file=DEFAULT_OUTPUT_FILE):
    input_file = Path(input_file)
    output_file = Path(output_file)

    if not input_file.exists():
        raise FileNotFoundError(f"Input coordinate workbook not found: {input_file}")

    np.random.seed(42)
    xls = pd.ExcelFile(input_file)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for output_sheet, source_sheet in SHEET_MAPPING.items():
            df = pd.read_excel(xls, sheet_name=source_sheet)
            df.columns = df.columns.str.strip().str.lower()

            coord_cols = [col for col in df.columns if "coord" in col]
            if not coord_cols:
                raise ValueError(
                    f"No coordinate column found in sheet '{source_sheet}'. "
                    f"Columns available: {df.columns.tolist()}"
                )

            coord_col = coord_cols[0]
            coords = (
                df[coord_col]
                .astype(str)
                .str.extract(r"(?P<lat>-?\d+\.?\d*),\s*(?P<lon>-?\d+\.?\d*)")
                .astype(float)
            )

            displaced = coords.apply(
                lambda row: displace_point(row["lat"], row["lon"], 10, 20),
                axis=1,
            )

            df["lat_private"] = [coord[0] for coord in displaced]
            df["lon_private"] = [coord[1] for coord in displaced]
            df.to_excel(writer, sheet_name=output_sheet, index=False)

    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    create_private_coordinates()
