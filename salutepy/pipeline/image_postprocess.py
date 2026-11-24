"""
Porting delle chiamate a R `magick` presenti (identiche) in
plot_images_wkclimate_mol01gfs.r, plot_images_daymax.r,
plot_images_t_dayice.r, plot_images_day.r, plot_images_onlybio.r:

    LogoLammaCNR <- image_read(logoimage)
    tempimg <- image_read(<var>_<id>.png)
    tempimg <- image_border(tempimg, "white", "0x70")
    tempimg <- image_crop(tempimg, "580x700")
    tempimg <- image_composite(tempimg, LogoLammaCNR, offset = "+260+10")
    tempimg <- image_annotate(tempimg, textvar, size=.., font=.., ...)
    tempimg <- image_annotate(tempimg, run_annotation, size=.., font=.., ...)
    image_write(tempimg, path=<output>.png)

Usa Pillow al posto di ImageMagick/`magick`. La geometria (crop 580x700,
bordo 0x70, offset logo +260+10, posizioni testo) è riprodotta
identicamente ai valori del sorgente R.

*** NOTA SUI FONT ***
Il sorgente R usa i font di sistema "Palatino" e "Sans" (nomi logici
risolti da ImageMagick/fontconfig sul server di produzione). Qui il
default è DejaVu Serif (sostituto di Palatino) e DejaVu Sans, disponibili
su questo sistema — la resa visiva del testo potrebbe differire
leggermente dall'originale se sul server di produzione erano installati i
font esatti "Palatino"/"Sans". Font path configurabili via FONT_SERIF/
FONT_SANS se necessario.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"  # sostituto di "Palatino"
FONT_SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # sostituto di "Sans"


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def image_border_white(img: Image.Image, top_bottom_px: int = 70) -> Image.Image:
    """Equivalente di image_border(img, "white", "0x70"): bordo bianco solo sopra/sotto."""
    w, h = img.size
    out = Image.new("RGBA", (w, h + 2 * top_bottom_px), "white")
    out.paste(img, (0, top_bottom_px), img if img.mode == "RGBA" else None)
    return out


def image_crop_topleft(img: Image.Image, width: int = 580, height: int = 700) -> Image.Image:
    """Equivalente di image_crop(img, "580x700"): crop dall'angolo in alto a sinistra (gravity NorthWest)."""
    return img.crop((0, 0, width, height))


def image_composite_offset(base: Image.Image, overlay: Image.Image, offset: tuple[int, int]) -> Image.Image:
    """Equivalente di image_composite(base, overlay, offset="+x+y")."""
    base = base.copy()
    base.paste(overlay, offset, overlay if overlay.mode == "RGBA" else None)
    return base


def image_annotate(
    img: Image.Image,
    text: str,
    location: tuple[int, int],
    size: int,
    font_path: str = FONT_SANS,
    color: str = "black",
) -> Image.Image:
    """
    Equivalente di image_annotate(img, text, size=.., font=.., color=.., location="+x+y").
    Il parametro `weight` di R (peso del font) non ha un analogo diretto
    in Pillow senza font varianti dedicate: qui si usa direttamente la
    variante Bold del font scelto (coerente con l'uso di pesi 300-700 nel
    sorgente R, tutti abbastanza marcati da giustificare grassetto).
    """
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    font = _load_font(font_path, size)
    draw.text(location, text, fill=color, font=font)
    return img


def compose_biometeo_image(
    raw_png_path: str | Path,
    logo_path: str | Path,
    title_text: str,
    subtitle_text: str,
    output_path: str | Path,
    title_size: int = 21,
    subtitle_size: int = 13,
    title_location: tuple[int, int] = (60, 61),
    subtitle_location: tuple[int, int] = (55, 86),
    logo_offset: tuple[int, int] = (260, 10),
    crop_size: tuple[int, int] = (580, 700),
    border_top_bottom: int = 70,
    remove_source: bool = True,
) -> Path:
    """
    Pipeline di post-processing completa, equivalente al blocco "Image
    management" comune a tutti i plot_images_*.r. Prende il PNG grezzo
    prodotto da GrADS e produce l'immagine finale con bordo, crop, logo
    e annotazioni testuali.
    """
    raw_png_path = Path(raw_png_path)
    output_path = Path(output_path)

    with Image.open(raw_png_path) as img:
        img = img.convert("RGBA")
        img = image_border_white(img, border_top_bottom)
        img = image_crop_topleft(img, *crop_size)

        with Image.open(logo_path) as logo:
            logo = logo.convert("RGBA")
            img = image_composite_offset(img, logo, logo_offset)

        img = image_annotate(img, title_text, title_location, title_size, font_path=FONT_SERIF)
        img = image_annotate(img, subtitle_text, subtitle_location, subtitle_size, font_path=FONT_SANS)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.convert("RGB").save(output_path, format="PNG")

    if remove_source:
        raw_png_path.unlink(missing_ok=True)

    return output_path


def add_watermarks(
    png_path: str | Path,
    logo_wk_path: str | Path,
    disclaimer_path: str | Path,
) -> None:
    """
    Porting di plot_logo_wkclimate.r / plot_logo_wkclimate_onlybio.r
    (le due chiamate `composite` di ImageMagick applicate in sequenza a
    ogni PNG già prodotto):

        composite -gravity NorthWest -geometry +58+8  <logoimagewk> <file> temp.png
        composite -gravity South     -geometry +30-10 <discimagewk> temp.png temp2.png

    Modifica il file in-place (stesso comportamento di file.rename in R).
    """
    png_path = Path(png_path)

    with Image.open(png_path) as base:
        base = base.convert("RGBA")

        with Image.open(logo_wk_path) as logo:
            logo = logo.convert("RGBA")
            base.paste(logo, (58, 8), logo)  # gravity NorthWest, offset +58+8

        with Image.open(disclaimer_path) as disc:
            disc = disc.convert("RGBA")
            w, h = base.size
            dw, dh = disc.size
            # gravity South, geometry +30-10: 30px orizzontale (ignorato per gravity South
            # pura, centrato), 10px di margine dal bordo inferiore
            x = (w - dw) // 2
            y = h - dh - 10
            base.paste(disc, (x, y), disc)

        base.convert("RGB").save(png_path, format="PNG")
