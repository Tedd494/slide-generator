"""
AI Slide-to-Image Generator
============================
A single-file Streamlit application that turns raw text content into a
polished, presentation-ready slide image (1920x1080) using:

  - OpenAI GPT-4 to summarize raw content into concise bullet points
  - OpenAI DALL-E 3 to generate an explanatory illustration
  - Pillow (PIL) to composite a background template, company logo,
    wrapped/summarized text, and the AI-generated image into one PNG.

Run with:  streamlit run app.py
"""

import io
import textwrap
from typing import List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

CANVAS_W, CANVAS_H = 1920, 1080
MARGIN = 60
DEFAULT_BG_COLOR = (24, 26, 35)

FONT_CANDIDATES_REGULAR = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
    "DejaVuSans.ttf",
    "Arial.ttf",
]
FONT_CANDIDATES_BOLD = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
    "DejaVuSans-Bold.ttf",
    "Arial-Bold.ttf",
]

LOGO_POSITIONS = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
LAYOUT_TYPES = [
    "Text Left & Image Right",
    "Text Right & Image Left",
    "Text Only (Full Width)",
    "Image Only (Full Bleed)",
]


# --------------------------------------------------------------------------
# OpenAI helpers
# --------------------------------------------------------------------------

def get_openai_client(api_key: str):
    """Instantiate an OpenAI client, or None if unavailable."""
    if not api_key:
        return None
    if OpenAI is None:
        st.error("The `openai` package is not installed. Run: pip install openai")
        return None
    return OpenAI(api_key=api_key)


def summarize_content(client, raw_text: str, title_hint: str = "") -> List[str]:
    """Call GPT-4 to turn raw slide content into 3-4 concise bullet points."""
    system_prompt = (
        "You are an expert presentation writer. Convert the user's raw notes "
        "into exactly 3 to 4 short, punchy bullet points suitable for a "
        "slide. Each bullet must be under 12 words, no markdown symbols, "
        "no numbering, and no trailing punctuation. Return ONLY the bullet "
        "lines, one per line, nothing else."
    )
    user_prompt = raw_text
    if title_hint:
        user_prompt = f"Slide title: {title_hint}\n\nRaw content:\n{raw_text}"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
        max_tokens=300,
    )
    raw_summary = response.choices[0].message.content.strip()

    bullets = []
    for line in raw_summary.splitlines():
        line = line.strip().lstrip("-•*0123456789. ").strip()
        if line:
            bullets.append(line)
    return bullets[:4] if bullets else [raw_summary]


def generate_dalle_image(client, prompt: str, size: str = "1024x1024") -> Image.Image:
    """Call DALL-E 3 to generate an explanatory illustration and return a PIL Image."""
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url
    img_bytes = requests.get(image_url, timeout=60).content
    return Image.open(io.BytesIO(img_bytes)).convert("RGBA")


# --------------------------------------------------------------------------
# Pillow / compositing helpers
# --------------------------------------------------------------------------

def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a TrueType font from common system locations, falling back gracefully."""
    candidates = FONT_CANDIDATES_BOLD if bold else FONT_CANDIDATES_REGULAR
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    # Last resort: Pillow's built-in bitmap font (Pillow >= 10 allows a size arg)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def wrap_text_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont,
                        max_width: int) -> List[str]:
    """Word-wrap `text` so every rendered line fits within `max_width` pixels."""
    words = text.split()
    if not words:
        return [""]

    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        w = draw.textlength(candidate, font=font)
        if w <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def fit_background(bg: Optional[Image.Image]) -> Image.Image:
    """Resize/crop an uploaded background to exactly fill the canvas; else a solid fallback."""
    if bg is None:
        return Image.new("RGB", (CANVAS_W, CANVAS_H), DEFAULT_BG_COLOR)
    bg = bg.convert("RGB")
    return ImageOps.fit(bg, (CANVAS_W, CANVAS_H), method=Image.LANCZOS, centering=(0.5, 0.5))


def paste_logo(canvas: Image.Image, logo: Optional[Image.Image], position: str) -> None:
    """Resize the logo proportionally and paste it into the requested corner."""
    if logo is None:
        return
    logo = logo.convert("RGBA")

    max_w = int(CANVAS_W * 0.14)
    max_h = int(CANVAS_H * 0.14)
    logo.thumbnail((max_w, max_h), Image.LANCZOS)
    lw, lh = logo.size

    coords = {
        "Top-Left": (MARGIN, MARGIN),
        "Top-Right": (CANVAS_W - lw - MARGIN, MARGIN),
        "Bottom-Left": (MARGIN, CANVAS_H - lh - MARGIN),
        "Bottom-Right": (CANVAS_W - lw - MARGIN, CANVAS_H - lh - MARGIN),
    }
    x, y = coords.get(position, coords["Top-Left"])
    canvas.paste(logo, (x, y), logo)


def compute_layout_rects(layout_type: str, logo_position: str) -> Tuple[Optional[Tuple[int, int, int, int]],
                                                                          Optional[Tuple[int, int, int, int]]]:
    """
    Return (text_rect, image_rect) as (x0, y0, x1, y1) tuples based on layout type.
    Either rect may be None if that element isn't used by the layout.
    """
    top_inset = MARGIN + (int(CANVAS_H * 0.14) if "Top" in logo_position else 0) + 20
    bottom_inset = MARGIN + (int(CANVAS_H * 0.14) if "Bottom" in logo_position else 0)

    top = max(MARGIN, top_inset)
    bottom = CANVAS_H - max(MARGIN, bottom_inset)
    gutter = 40

    if layout_type == "Text Left & Image Right":
        half = CANVAS_W // 2
        text_rect = (MARGIN, top, half - gutter, bottom)
        image_rect = (half + gutter, top, CANVAS_W - MARGIN, bottom)
    elif layout_type == "Text Right & Image Left":
        half = CANVAS_W // 2
        image_rect = (MARGIN, top, half - gutter, bottom)
        text_rect = (half + gutter, top, CANVAS_W - MARGIN, bottom)
    elif layout_type == "Text Only (Full Width)":
        text_rect = (MARGIN, top, CANVAS_W - MARGIN, bottom)
        image_rect = None
    elif layout_type == "Image Only (Full Bleed)":
        text_rect = None
        image_rect = (MARGIN, top, CANVAS_W - MARGIN, bottom)
    else:
        text_rect = (MARGIN, top, CANVAS_W - MARGIN, bottom)
        image_rect = None

    return text_rect, image_rect


def fit_text_block(draw: ImageDraw.ImageDraw, title: str, bullets: List[str],
                    rect: Tuple[int, int, int, int], base_font_size: int,
                    min_font_size: int = 16):
    """
    Auto-shrink the font size until the title + wrapped bullets fit inside `rect`.
    Returns (title_font, body_font, wrapped_bullet_lines, line_height, title_height).
    """
    x0, y0, x1, y1 = rect
    box_w = x1 - x0
    box_h = y1 - y0

    font_size = base_font_size
    while font_size >= min_font_size:
        title_font = load_font(font_size + 14, bold=True)
        body_font = load_font(font_size, bold=False)

        title_lines = wrap_text_to_width(draw, title, title_font, box_w) if title else []
        title_line_height = (title_font.size if hasattr(title_font, "size") else font_size + 14) * 1.3
        title_height = len(title_lines) * title_line_height + (30 if title_lines else 0)

        line_height = font_size * 1.5
        bullet_gap = font_size * 0.6
        all_bullet_lines = []
        total_bullet_height = 0
        for bullet in bullets:
            wrapped = wrap_text_to_width(draw, f"\u25CF  {bullet}", body_font, box_w)
            all_bullet_lines.append(wrapped)
            total_bullet_height += len(wrapped) * line_height + bullet_gap

        total_height = title_height + total_bullet_height
        if total_height <= box_h or font_size <= min_font_size:
            return title_font, body_font, title_lines, all_bullet_lines, line_height, title_height, bullet_gap
        font_size -= 2

    # Should not reach here, but keep a safe fallback
    return title_font, body_font, title_lines, all_bullet_lines, line_height, title_height, bullet_gap


def draw_text_panel(canvas: Image.Image, rect: Tuple[int, int, int, int], title: str,
                     bullets: List[str], font_size: int, text_color: str,
                     panel_backdrop: bool = True) -> None:
    """Draw a semi-transparent backdrop (optional) plus the title and wrapped bullets."""
    if rect is None or (not title and not bullets):
        return

    x0, y0, x1, y1 = rect
    draw = ImageDraw.Draw(canvas)

    (title_font, body_font, title_lines, all_bullet_lines,
     line_height, title_height, bullet_gap) = fit_text_block(draw, title, bullets, rect, font_size)

    if panel_backdrop:
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        odraw.rounded_rectangle(
            [x0 - 24, y0 - 24, x1 + 24, y1 + 24], radius=24, fill=(0, 0, 0, 90)
        )
        canvas.paste(Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB"), (0, 0))
        draw = ImageDraw.Draw(canvas)

    title_line_height = (title_font.size if hasattr(title_font, "size") else font_size + 14) * 1.3
    cursor_y = y0
    for line in title_lines:
        draw.text((x0, cursor_y), line, font=title_font, fill=text_color)
        cursor_y += title_line_height
    if title_lines:
        cursor_y += 20

    for wrapped in all_bullet_lines:
        for i, line in enumerate(wrapped):
            draw.text((x0, cursor_y), line, font=body_font, fill=text_color)
            cursor_y += line_height
        cursor_y += bullet_gap


def paste_generated_image(canvas: Image.Image, gen_image: Optional[Image.Image],
                           rect: Optional[Tuple[int, int, int, int]]) -> None:
    """Fit the DALL-E illustration (preserving aspect ratio) inside its placeholder rect."""
    if gen_image is None or rect is None:
        return
    x0, y0, x1, y1 = rect
    box_w, box_h = x1 - x0, y1 - y0

    img = gen_image.convert("RGBA").copy()
    img.thumbnail((box_w, box_h), Image.LANCZOS)
    iw, ih = img.size

    px = x0 + (box_w - iw) // 2
    py = y0 + (box_h - ih) // 2

    # subtle rounded backdrop behind the image
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rounded_rectangle([px - 14, py - 14, px + iw + 14, py + ih + 14], radius=20, fill=(255, 255, 255, 30))
    canvas.paste(Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB"), (0, 0))

    canvas.paste(img, (px, py), img)


def compose_slide(background: Optional[Image.Image], logo: Optional[Image.Image],
                   gen_image: Optional[Image.Image], title: str, bullets: List[str],
                   font_size: int, text_color: str, logo_position: str,
                   layout_type: str) -> Image.Image:
    """Top-level compositor: builds the final 1920x1080 slide."""
    canvas = fit_background(background)
    text_rect, image_rect = compute_layout_rects(layout_type, logo_position)

    if layout_type != "Image Only (Full Bleed)":
        draw_text_panel(canvas, text_rect, title, bullets, font_size, text_color)

    paste_generated_image(canvas, gen_image, image_rect)
    paste_logo(canvas, logo, logo_position)

    return canvas


# --------------------------------------------------------------------------
# Streamlit App
# --------------------------------------------------------------------------

def init_session_state():
    defaults = {
        "summary_bullets": [],
        "generated_image": None,
        "final_slide": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main():
    st.set_page_config(page_title="AI Slide-to-Image Generator", layout="wide")
    init_session_state()

    st.title("🖼️ AI Slide-to-Image Generator")
    st.caption("Turn raw notes into a polished 1920x1080 slide using GPT-4, DALL-E 3, and Pillow.")

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("🔑 API Configuration")
        openai_api_key = st.text_input("OpenAI API Key", type="password",
                                        help="Used for both GPT-4 summarization and DALL-E 3 image generation.")

        st.divider()
        st.header("🎨 Brand Assets")
        logo_file = st.file_uploader("Company Logo", type=["png", "jpg", "jpeg", "webp"])
        bg_file = st.file_uploader("Background Template", type=["png", "jpg", "jpeg", "webp"])

        logo_image = Image.open(logo_file) if logo_file else None
        bg_image = Image.open(bg_file) if bg_file else None

        if logo_image:
            st.image(logo_image, caption="Logo preview", width=120)
        if bg_image:
            st.image(bg_image, caption="Background preview", width=180)

        st.divider()
        st.header("⚙️ Layout Settings")
        layout_type = st.selectbox("Layout Type", LAYOUT_TYPES, index=0)
        logo_position = st.selectbox("Logo Position", LOGO_POSITIONS, index=1)
        font_size = st.slider("Base Font Size", min_value=20, max_value=90, value=42, step=2)
        text_color = st.color_picker("Text Color", value="#FFFFFF")

    client = get_openai_client(openai_api_key)

    # ---------------- Main: Content Input ----------------
    st.subheader("1. Slide Content")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        slide_title = st.text_input("Slide Title", value="")
        raw_content = st.text_area("Raw Slide Content", height=220,
                                    placeholder="Paste your raw notes, meeting minutes, or report text here...")
        summarize_clicked = st.button("✨ Summarize Content using AI", use_container_width=True)

    with col_b:
        st.markdown("**Summarized Bullet Points**")
        if summarize_clicked:
            if not client:
                st.error("Please provide a valid OpenAI API key in the sidebar.")
            elif not raw_content.strip():
                st.warning("Please enter some raw content to summarize first.")
            else:
                with st.spinner("Calling GPT-4 to summarize..."):
                    try:
                        st.session_state.summary_bullets = summarize_content(client, raw_content, slide_title)
                    except Exception as e:
                        st.error(f"GPT-4 summarization failed: {e}")

        bullets_text = st.text_area(
            "Edit bullets if needed (one per line)",
            value="\n".join(st.session_state.summary_bullets),
            height=220,
        )
        st.session_state.summary_bullets = [b.strip() for b in bullets_text.splitlines() if b.strip()]

    st.divider()

    # ---------------- Main: Image Generation ----------------
    st.subheader("2. Explanatory Illustration (DALL-E 3)")
    col_c, col_d = st.columns([2, 1])
    with col_c:
        image_prompt = st.text_input(
            "Image Generation Prompt",
            placeholder="e.g. A minimalist flat illustration of a team collaborating around a growth chart",
        )
        dalle_size = st.selectbox("Image Size", ["1024x1024", "1792x1024", "1024x1792"], index=0)
        generate_img_clicked = st.button("🎨 Generate Illustration with DALL-E 3", use_container_width=True)

    with col_d:
        if generate_img_clicked:
            if not client:
                st.error("Please provide a valid OpenAI API key in the sidebar.")
            elif not image_prompt.strip():
                st.warning("Please enter an image generation prompt first.")
            else:
                with st.spinner("Calling DALL-E 3..."):
                    try:
                        st.session_state.generated_image = generate_dalle_image(client, image_prompt, dalle_size)
                    except Exception as e:
                        st.error(f"DALL-E 3 generation failed: {e}")

        if st.session_state.generated_image is not None:
            st.image(st.session_state.generated_image, caption="Generated illustration", use_container_width=True)

    st.divider()

    # ---------------- Main: Compose & Output ----------------
    st.subheader("3. Compose Final Slide")
    if st.button("🖌️ Generate Composited Slide", type="primary", use_container_width=True):
        try:
            final_slide = compose_slide(
                background=bg_image,
                logo=logo_image,
                gen_image=st.session_state.generated_image,
                title=slide_title,
                bullets=st.session_state.summary_bullets,
                font_size=font_size,
                text_color=text_color,
                logo_position=logo_position,
                layout_type=layout_type,
            )
            st.session_state.final_slide = final_slide
        except Exception as e:
            st.error(f"Failed to compose slide: {e}")

    if st.session_state.final_slide is not None:
        st.image(st.session_state.final_slide, caption="Final Slide (1920x1080)", use_container_width=True)

        buf = io.BytesIO()
        st.session_state.final_slide.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Slide as PNG",
            data=buf.getvalue(),
            file_name="ai_generated_slide.png",
            mime="image/png",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
