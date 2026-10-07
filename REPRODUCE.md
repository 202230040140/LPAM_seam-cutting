# LPAM: original C++ kernels with Python orchestration

This port implements **Homography + baseline seam-cutting + LPAM**, the
configuration covered by Table 2 of the ICCV 2025 paper. It uses CPU inference
and the existing obj-gsp-sam Python environment for image processing and pyiqa.
No learned weights or depth maps are required.

The original Dense SIFT, BPFlow and maxflow source algorithms are built into
three independent DLLs. Generated build copies disable MATLAB/Qt image I/O;
original C++ source files remain unchanged. Run `python -m lpam_py.build_native`
with Visual Studio x64 C++ tools installed. Outputs are ignored under
`lpam_py/_native`, outside benchmark result roots.

The Python frontend uses OpenCV SIFT, unit-normalized SSD matching (10% of the
SSD range and ratio 0.6), normalized DLT/RANSAC (seed 0, up to 2000 trials,
threshold 0.01), Gaussian SSIM, Otsu marking, localized original SIFT Flow and
sigmoid displacement with beta 8. Patches use size 21, quality factor 1.5.
The returned locally realigned images are used in final composition; the
upstream demo originally combined old images with the new seam. The local
main.m composition line is corrected to use warped_img1_/warped_img2_, too.
MATLAB is unavailable on this machine, so this demo correction is not runtime-
verified; all benchmark images come from the validated Python/C++ port.
Final DLT repeats the inlier normalization and VGG per-axis sample-standard-
deviation conditioning from calcHomo/homography_fit. A failed global homography
(including a projective horizon through the source image) is recorded, not
replaced. Known weak-scene smoke failures do not suppress the remaining attempts;
all three datasets must nevertheless have a successful smoke before full runs.

OpenCV feature detection/initial bilinear warp, NumPy random sampling, Keys
bicubic sampling/resize, edge handling and canvas rounding can differ from
MATLAB. Numerical identity to the MATLAB implementation has not been verified.
There is no alternative flow or homography-only fallback after LPAM errors.
Singleton coarse pyramid axes use zero offsets (both patches have equal size).

From MethodManagement, use `py -3.13 -m methodman run LPAM --command all`.
Individual commands: build, smoke, stitchbench_smoke, stitchbench, hd3d_smoke,
hd3d, lpsd_smoke, lpsd, audit. Audit rechecks all 214 input/parameter/implementation
hashes, final-image dimensions and hashes, cleanup, metrics and preserved methods.
The batch interface is
`python -m lpam_py.run --benchmark stitchbench|hd3d|lpsd` with optional
`--manifest`, `--result-root`, `--scene`, `--limit`, `--device`,
`--max-input-edge`, `--max-out-height`, `--max-canvas-pixels`, `--timeout`,
`--lpips-max-side`, `--skip-existing`, `--force`, `--stop-on-error`.

Input longest edge is capped at 2048 by default. Only allocation/canvas errors
trigger 1536/1024 retries; canvas height is at most 8000 and area at most 16M.
Each inference process has a 600-second timeout. Inference and metric success
are separate states. Successful cache reuse requires matching inputs, fixed
parameters, original kernels/frontend/evaluator version and output hashes.

Each pair stores **only lpam/stitch_result.png**. HD3D/LPS-D records are central
in `_global_work/lpam`; StitchBench records are in existing run_metadata.json.
Warped images, seams, descriptors and flow remain in memory. GT alignment
images/masks and request/status files use TemporaryDirectory with finally cleanup.
StitchBench evaluates the entire final canvas with pyiqa NIQE and reports native
MDR as N/A. GT benchmarks reuse the existing evaluator for GT-alignment MDR,
NIQE, PSNR, SSIM, LPIPS and RMSE; these MDR protocols must not be conflated.
Existing other-method rows and Depth-GSP parameter/gate metadata are preserved.
Preservation audits compare other-method metric rows and final-image SHA256
values. The all command attempts every manifest pair even when genuine failures
are recorded, then exits nonzero if any failures remain; this does not mean
later datasets were skipped. Inspect the centralized attempt records/reports.

Validation includes graph-cut labels, a known SIFT Flow translation, local
boundary constraints, safe-region invariance, final use of realigned pixels,
normalized homography, verified cache reuse, and owned-process-tree timeout
cleanup. The bundled image pair must activate local repair. No intermediate
benchmark images are saved during these tests.

## Validated run (2026-10-07)

All 214 manifest pairs were attempted with fixed parameters. Final-image and
complete-metric counts were 99/100 on StitchBench, 77/78 on HD3D, and 36/36 on
LPS-D: **212 final images / 212 complete metric records**. The 214-image target
was not fully achieved; genuine failures were retained without substitution:

- StitchBench OBJ-GSP-tree2: projective horizon crosses the target image.
- HD3D Outdoor_002/pair_23: canvas exceeds limits at all three input caps
  (2048, 1536, 1024); final attempted canvas was 5854x4362, above 16M pixels.

The combined audit passed for all 214 input/configuration/core-version hashes,
image checksums/dimensions, metrics, final-only layout, 20 comparison panels,
unchanged other-method images/rows, and preserved DDAF metadata. Eight native/
runner tests and two targeted-report tests passed. No LPAM pair work directory
or inference process remained after completion. MATLAB identity is unverified.
