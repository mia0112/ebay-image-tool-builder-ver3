from __future__ import annotations

import io
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

from .api_bg_remove import BackgroundRemovalAPIClient, BackgroundRemovalError
from .config import AppConfig
from .frame_composer import FitResult, compose_on_canvas
from .gemini_client import GeminiVisionClient, GeminiVisionError
from .mask_cleaner import (
    guided_opencv_mask,
    load_image_rgb,
    make_preview,
    plain_opencv_remove,
    refine_alpha,
    rgba_from_mask,
    scale_analysis,
    trim_uniform_border,
    validate_analysis,
)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class ImageResult:
    source_file: str
    output_file: str
    method: str
    api_cost: float
    status: str
    note: str = ""


@dataclass
class FolderResult:
    folder_name: str
    total_images: int
    api_calls: int
    estimated_cost: float
    status: str
    images: List[ImageResult]


def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS])


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _save_report(folder: Path, report: FolderResult, output_subfolder: str = "output") -> None:
    report_path = folder / output_subfolder / "_processing_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")


def _build_gemini_client(config: AppConfig, log: Callable[[str], None]) -> Optional[GeminiVisionClient]:
    if not config.gemini_enabled:
        return None
    api_key = GeminiVisionClient.api_key_from_env_or_value(config.gemini_api_key)
    if not api_key:
        log("Gemini disabled for this run: CONFIG/gemini_api_key.txt is empty and GEMINI_API_KEY/GOOGLE_API_KEY env is not set.")
        return None
    try:
        return GeminiVisionClient(
            api_key=api_key,
            model=config.gemini_model,
            retry_model=config.gemini_retry_model,
            timeout_seconds=config.gemini_timeout_seconds,
        )
    except GeminiVisionError as exc:
        log(f"Gemini disabled for this run: {exc}")
        return None


def _build_removebg_client(config: AppConfig) -> Optional[BackgroundRemovalAPIClient]:
    if not config.api_endpoint or not config.api_key:
        return None
    return BackgroundRemovalAPIClient(
        endpoint=config.api_endpoint,
        api_key=config.api_key,
        auth_header_name=config.auth_header_name,
        image_field_name=config.image_field_name,
        extra_form_fields=config.extra_form_fields,
    )


def _component_mode_from_analysis(config: AppConfig, analysis: Optional[Dict]) -> str:
    if config.component_mode in {"single_part", "multi_part"}:
        return config.component_mode
    if analysis and analysis.get("product_type") == "single_object":
        return "single_part"
    return "multi_part"


def _remove_with_gemini_guided_opencv(
    image: Image.Image,
    config: AppConfig,
    gemini_client: GeminiVisionClient,
    debug_dir: Optional[Path],
    stem: str,
) -> Tuple[Image.Image, str, Dict]:
    preview, sx, sy = make_preview(image, config.gemini_preview_max_size)
    if debug_dir:
        preview.save(debug_dir / f"{stem}_01_preview.jpg", quality=92)

    raw_analysis, used_model = gemini_client.analyze_with_retry(
        preview,
        threshold=config.gemini_confidence_threshold,
        retry_on_low_confidence=config.gemini_retry_on_low_confidence,
    )
    analysis = scale_analysis(raw_analysis, sx, sy, image.width, image.height)
    ok, reason = validate_analysis(analysis, image.width, image.height, config.gemini_confidence_threshold)
    if debug_dir:
        _save_json(debug_dir / f"{stem}_02_gemini_analysis.json", {"model": used_model, "raw": raw_analysis, "scaled": analysis, "valid": ok, "reason": reason})
    if not ok:
        raise GeminiVisionError(reason)

    # v3: OpenCV only edits the alpha mask. Do not inpaint or alter RGB pixels
    # before masking; Gemini remove_regions are applied directly to alpha.
    if debug_dir:
        image.save(debug_dir / f"{stem}_03_mask_source_rgb_unchanged.jpg", quality=95)

    component_mode = _component_mode_from_analysis(config, analysis)
    mask = guided_opencv_mask(
        image,
        analysis,
        threshold=config.local_bg_threshold,
        bbox_expand_ratio=config.opencv_bbox_expand_ratio,
        component_mode=component_mode,
        keep_small_accessories=config.keep_small_accessories,
    )
    rgba = rgba_from_mask(image, mask)
    rgba = refine_alpha(rgba, analysis=analysis, component_mode=component_mode, keep_small_accessories=config.keep_small_accessories)
    if debug_dir:
        Image.fromarray(mask).save(debug_dir / f"{stem}_04_mask.png")
        rgba.save(debug_dir / f"{stem}_05_cutout.png")
    note = f"gemini_guided_opencv; model={used_model}; product_type={analysis.get('product_type')}; confidence={float(analysis.get('confidence') or 0):.2f}"
    return rgba, note, analysis


def _retry_leftover_text_cleanup(
    image: Image.Image,
    rgba: Image.Image,
    config: AppConfig,
    gemini_client: Optional[GeminiVisionClient],
    debug_dir: Optional[Path],
    stem: str,
) -> Tuple[Image.Image, str, Optional[Dict]]:
    if gemini_client is None:
        return rgba, "", None
    preview, sx, sy = make_preview(image, config.gemini_preview_max_size)
    raw_analysis, used_model = gemini_client.analyze_strict_sweep(preview)
    analysis = scale_analysis(raw_analysis, sx, sy, image.width, image.height)
    ok, reason = validate_analysis(analysis, image.width, image.height, config.gemini_confidence_threshold)
    if debug_dir:
        _save_json(debug_dir / f"{stem}_06b_gemini_strict_sweep.json", {"model": used_model, "raw": raw_analysis, "scaled": analysis, "valid": ok, "reason": reason})
    if not ok:
        raise GeminiVisionError(f"strict text sweep invalid: {reason}")
    component_mode = _component_mode_from_analysis(config, analysis)
    mask = guided_opencv_mask(
        image,
        analysis,
        threshold=config.local_bg_threshold,
        bbox_expand_ratio=config.opencv_bbox_expand_ratio,
        component_mode=component_mode,
        keep_small_accessories=config.keep_small_accessories,
    )
    rerun = rgba_from_mask(image, mask)
    rerun = refine_alpha(rerun, analysis=analysis, component_mode=component_mode, keep_small_accessories=config.keep_small_accessories)
    if debug_dir:
        Image.fromarray(mask).save(debug_dir / f"{stem}_06c_strict_sweep_mask.png")
        rerun.save(debug_dir / f"{stem}_06d_strict_sweep_cutout.png")
    return rerun, f"strict_text_sweep; model={used_model}", analysis


def _remove_with_api(
    image: Image.Image,
    img_name: str,
    config: AppConfig,
    api_client: Optional[BackgroundRemovalAPIClient],
    current_cost: float,
    budget: float,
    analysis: Optional[Dict],
) -> Tuple[Image.Image, float, str]:
    if api_client is None:
        raise RuntimeError("remove.bg API is not configured.")
    next_cost = current_cost + config.cost_per_api_call
    if next_cost > budget:
        raise RuntimeError(
            f"API fallback blocked by cost guard. Current cost={current_cost:.2f}, next would be {next_cost:.2f}, budget={budget:.2f}"
        )
    rgba = api_client.remove_background(_image_to_png_bytes(image), filename=img_name)
    component_mode = _component_mode_from_analysis(config, analysis)
    rgba = refine_alpha(rgba, analysis=analysis, component_mode=component_mode, keep_small_accessories=config.keep_small_accessories)
    return rgba, config.cost_per_api_call, f"removebg_api; refined={component_mode}"


def _remove_with_plain_opencv(image: Image.Image, config: AppConfig) -> Tuple[Image.Image, str]:
    mode = "multi_part" if config.component_mode == "auto" else config.component_mode
    rgba = plain_opencv_remove(image, threshold=config.local_bg_threshold, component_mode=mode)
    rgba = refine_alpha(rgba, analysis=None, component_mode=mode, keep_small_accessories=True)
    return rgba, f"opencv_local_fallback; component_mode={mode}"


def _background_remove(
    image: Image.Image,
    img_name: str,
    config: AppConfig,
    gemini_client: Optional[GeminiVisionClient],
    api_client: Optional[BackgroundRemovalAPIClient],
    current_cost: float,
    budget: float,
    debug_dir: Optional[Path],
    stem: str,
) -> Tuple[Image.Image, str, float, Optional[Dict]]:
    analysis: Optional[Dict] = None
    errors: List[str] = []

    if config.remove_background_flow == "gemini_guided_opencv" and gemini_client is not None:
        try:
            rgba, note, analysis = _remove_with_gemini_guided_opencv(image, config, gemini_client, debug_dir, stem)
            return rgba, note, 0.0, analysis
        except Exception as exc:
            errors.append(f"Gemini/OpenCV failed: {exc}")

    if config.remove_background_flow == "removebg_api":
        rgba, api_cost, note = _remove_with_api(image, img_name, config, api_client, current_cost, budget, analysis)
        return rgba, note, api_cost, analysis

    # Fallback chain.
    if config.fallback_mode in {"removebg_api", "removebg_api_then_opencv"}:
        try:
            rgba, api_cost, note = _remove_with_api(image, img_name, config, api_client, current_cost, budget, analysis)
            if errors:
                note += "; " + " | ".join(errors)
            return rgba, note, api_cost, analysis
        except Exception as exc:
            errors.append(f"remove.bg fallback failed: {exc}")
            if config.fallback_mode == "removebg_api":
                raise RuntimeError(" | ".join(errors))

    if config.fallback_mode in {"opencv_local", "removebg_api_then_opencv"} or config.local_bg_enabled:
        rgba, note = _remove_with_plain_opencv(image, config)
        if errors:
            note += "; " + " | ".join(errors)
        return rgba, note, 0.0, analysis

    raise RuntimeError(" | ".join(errors) if errors else "No background-removal method is available.")


def _compose_final(
    rgba: Image.Image,
    frame_image: Optional[Image.Image],
    config: AppConfig,
) -> FitResult:
    return compose_on_canvas(
        rgba,
        canvas_size=config.canvas_size,
        product_max_size=config.product_max_size,
        frame_image=frame_image,
        frame_safe_margin_left=config.frame_safe_margin_left,
        frame_safe_margin_top=config.frame_safe_margin_top,
        frame_safe_margin_right=config.frame_safe_margin_right,
        frame_safe_margin_bottom=config.frame_safe_margin_bottom,
        safe_area_padding_ratio=config.safe_area_padding_ratio,
        auto_fit_to_frame=config.auto_fit_to_frame,
        target_product_width_ratio=config.target_product_width_ratio,
        target_product_height_ratio=config.target_product_height_ratio,
        max_product_width_ratio=config.max_product_width_ratio,
        max_product_height_ratio=config.max_product_height_ratio,
        frame_collision_check=config.frame_collision_check,
        frame_clearance_px=config.frame_clearance_px,
        scale_down_step=config.scale_down_step,
        max_fit_attempts=config.max_fit_attempts,
        allowed_frame_overlap_ratio=config.allowed_frame_overlap_ratio,
    )


def process_product_folder(
    folder: Path,
    config: AppConfig,
    log: Callable[[str], None],
) -> FolderResult:
    images = list_images(folder)
    output_dir = folder / config.output_subfolder_name if config.create_output_subfolder else folder
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_root = output_dir / "debug" if config.save_debug_files else None
    if debug_root:
        debug_root.mkdir(parents=True, exist_ok=True)

    frame_image = None
    if config.frame_file:
        frame_path = Path(config.frame_file)
        if frame_path.exists():
            frame_image = Image.open(frame_path).convert("RGBA")

    api_calls = 0
    total_cost = 0.0
    results: List[ImageResult] = []
    budget = config.current_budget()
    gemini_client = _build_gemini_client(config, log)
    api_client = _build_removebg_client(config)

    if not images:
        report = FolderResult(folder.name, 0, 0, 0.0, "error", [])
        _save_report(folder, report, config.output_subfolder_name)
        return report

    log(f"\nProcessing folder: {folder.name}")
    for img_path in images:
        try:
            image = load_image_rgb(img_path)
            if getattr(config, "source_trim_uniform_border", False):
                image = trim_uniform_border(image)
            debug_dir = debug_root / img_path.stem if debug_root else None
            if debug_dir:
                debug_dir.mkdir(parents=True, exist_ok=True)
                image.save(debug_dir / f"{img_path.stem}_00_input_rgb_unchanged.jpg", quality=95)

            rgba, remove_note, api_cost, analysis = _background_remove(
                image,
                img_path.name,
                config,
                gemini_client,
                api_client,
                total_cost,
                budget,
                debug_dir,
                img_path.stem,
            )
            if api_cost:
                api_calls += 1
                total_cost += api_cost

            fit = _compose_final(rgba, frame_image, config)
            final_image = fit.image

            qa_note = ""
            if config.qa_after_compose and gemini_client is not None:
                try:
                    qa = gemini_client.qa(final_image)
                    if debug_dir:
                        _save_json(debug_dir / f"{img_path.stem}_06_qa.json", qa)
                    qa_note = f"; qa_pass={bool(qa.get('pass', False))}"
                    # A practical retry for frame overlap only. Source-mask retry for leftover text
                    # should be handled by the first Gemini analysis pass.
                    if config.qa_retry_once and not bool(qa.get("pass", False)):
                        issue_types = {str(i.get("type", "")) for i in qa.get("issues", []) if isinstance(i, dict)}
                        if "leftover_text" in issue_types:
                            try:
                                rgba_retry, retry_note, analysis_retry = _retry_leftover_text_cleanup(
                                    image, rgba, config, gemini_client, debug_dir, img_path.stem
                                )
                                fit = _compose_final(rgba_retry, frame_image, config)
                                final_image = fit.image
                                rgba = rgba_retry
                                if analysis_retry is not None:
                                    analysis = analysis_retry
                                qa_note += f"; retried_strict_text_sweep={retry_note}"
                            except Exception as exc:
                                qa_note += f"; strict_text_sweep_failed={exc}"
                        if "frame_overlap" in issue_types:
                            old_w, old_h = config.max_product_width_ratio, config.max_product_height_ratio
                            config.max_product_width_ratio *= 0.92
                            config.max_product_height_ratio *= 0.92
                            fit = _compose_final(rgba, frame_image, config)
                            final_image = fit.image
                            config.max_product_width_ratio, config.max_product_height_ratio = old_w, old_h
                            qa_note += "; retried_smaller_for_frame_overlap"
                except Exception as exc:
                    qa_note = f"; qa_skipped_error={exc}"

            if debug_dir:
                final_image.save(debug_dir / f"{img_path.stem}_07_final.jpg", quality=95)
                _save_json(debug_dir / f"{img_path.stem}_08_fit.json", asdict(fit))

            output_file = output_dir / f"{img_path.stem}_final.jpg"
            final_image.save(output_file, format="JPEG", quality=95)
            method = "api" if api_cost else ("gemini_opencv" if analysis else "opencv")
            note = f"{remove_note}; fit={fit.note}; scale={fit.scale:.3f}; frame_overlap={fit.overlap_ratio:.5f}{qa_note}"
            results.append(ImageResult(img_path.name, output_file.name, method, api_cost, "done", note))
            log(f"- {img_path.name}: OK → {output_file.name} [{method}] {note}")
        except (BackgroundRemovalError, Exception) as exc:
            results.append(ImageResult(img_path.name, "", "failed", 0.0, "error", str(exc)))
            log(f"- {img_path.name}: ERROR → {exc}")

    status = "done" if all(r.status == "done" for r in results) else "partial"
    report = FolderResult(folder.name, len(images), api_calls, round(total_cost, 4), status, results)
    _save_report(folder, report, config.output_subfolder_name)
    return report


def find_product_folders(root_folder: Path) -> List[Path]:
    images_in_root = list_images(root_folder)
    if images_in_root:
        return [root_folder]
    subfolders = [p for p in root_folder.iterdir() if p.is_dir() and not p.name.startswith(".") and p.name != "__MACOSX"]
    return [p for p in subfolders if list_images(p)]
