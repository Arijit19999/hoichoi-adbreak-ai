"""IAB VMAP 1.0 manifest with inline VAST 3.0 linear ads (one creative per mid-roll)."""

from lxml import etree

VMAP_NS = "http://www.iab.net/videosuite/vmap"
AD_SYSTEM = "hoichoi-adbreak-ai"


def _offset(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _duration(seconds: float) -> str:
    return _offset(seconds).split(".")[0]


def build_vmap(plan: dict, media_base_url: str) -> bytes:
    """media_base_url: absolute URL prefix the creative paths are served under (e.g. https://host/)."""
    base = media_base_url.rstrip("/") + "/"
    vmap = etree.Element(f"{{{VMAP_NS}}}VMAP", nsmap={"vmap": VMAP_NS}, version="1.0")

    for brk in plan["breaks"]:
        creative = brk["creative"]
        ad_break = etree.SubElement(vmap, f"{{{VMAP_NS}}}AdBreak",
                                    timeOffset=_offset(brk["time"]), breakType="linear", breakId=brk["id"])
        source = etree.SubElement(ad_break, f"{{{VMAP_NS}}}AdSource",
                                  id=f"{brk['id']}-source", allowMultipleAds="false", followRedirects="false")
        data = etree.SubElement(source, f"{{{VMAP_NS}}}VASTAdData")

        vast = etree.SubElement(data, "VAST", version="3.0")
        ad = etree.SubElement(vast, "Ad", id=f"{brk['brand_id']}-{creative['id']}", sequence="1")
        inline = etree.SubElement(ad, "InLine")
        etree.SubElement(inline, "AdSystem", version="1.0").text = AD_SYSTEM
        etree.SubElement(inline, "AdTitle").text = f"{brk['display_name']} - {creative['id']}"
        etree.SubElement(inline, "Description").text = (
            f"{brk['category']} | fit {brk['fit']:.2f} | {brk['reason']}")
        etree.SubElement(inline, "Impression", id="imp").text = etree.CDATA(
            f"{base}api/track?event=impression&break={brk['id']}")
        creatives = etree.SubElement(inline, "Creatives")
        cr = etree.SubElement(creatives, "Creative", id=creative["id"], sequence="1", AdID=creative["id"])
        linear = etree.SubElement(cr, "Linear")
        etree.SubElement(linear, "Duration").text = _duration(creative["duration_sec"])
        media_files = etree.SubElement(linear, "MediaFiles")
        media = etree.SubElement(media_files, "MediaFile", delivery="progressive", type="video/mp4",
                                 width="960", height="540", scalable="true", maintainAspectRatio="true")
        media.text = etree.CDATA(base + creative["url"].lstrip("/"))

        extensions = etree.SubElement(ad_break, f"{{{VMAP_NS}}}Extensions")
        ext = etree.SubElement(extensions, f"{{{VMAP_NS}}}Extension", type="hoichoi-adbreak-ai")
        etree.SubElement(ext, "BreakScore").text = f"{brk['score']:.3f}"
        etree.SubElement(ext, "Transition").text = str(brk.get("transition"))
        etree.SubElement(ext, "Scenes", before=str(brk["scene_before"]), after=str(brk["scene_after"]))

    return etree.tostring(vmap, xml_declaration=True, encoding="UTF-8", pretty_print=True)
