from pathlib import Path

from PIL import Image


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "build" / "desktop-resources"
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(root / "logos" / "TangoPosture_logo_under_1MB.png") as source:
        icon = source.convert("RGBA")
        icon.save(
            output / "TangoPosture.ico",
            sizes=[(size, size) for size in (16, 24, 32, 48, 64, 128, 256)],
        )
        icon.save(output / "TangoPosture.icns")
        icon.resize((256, 256), Image.Resampling.LANCZOS).save(output / "TangoPosture.png")
    print(f"Prepared macOS, Windows, and Linux icons in {output}")


if __name__ == "__main__":
    main()
