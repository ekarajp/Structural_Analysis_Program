# Sources of the built-in catalogs

Built by `scripts/build_catalog.py`; every tabulated row of the Thai standards is
checked against the area and moment of inertia computed from its dimensions.

| File | Content | Source |
|---|---|---|
| `steel_tis.csv` | TIS 1227-2558 (= JIS G3192): H, I-beams, channels, equal / unequal angles, T; TIS 1228-2561 (= JIS G3350): every annex table - angles, C, lipped C, Z, lipped Z, hats | Thai Industrial Standards, full texts published by TISI (service.tisi.go.th/fulltext/1227_2558.pdf, 1228_2561.pdf) |
| `steel_tis.csv` (TIS 107-2566) | Round tubes (JIS G3444 sizes), square and rectangular tubes (JIS G3466), and the ASTM A500 / EN 10219 sizes of the standard's annex tables | The TISI full text of TIS 107-2566 cannot be downloaded; its annex tables as printed in the Pacific Pipe structural catalogue 2026 (pacificpipe.co.th), every row checked against the area and I of its dimensions; the JIS rows checked against the TIS 107-2533 full text (ruengrawin.com/uploads/file/TIS107-2533.pdf) and JIS G3466 values in the NSMP hollow-section catalogue |
| `steel_gb.csv` | H (HW, HM, HN, HT: GB/T 11263-2017); I-beams, channels, equal / unequal angles (GB/T 706-2016) | GB/T 706-2016 full text (A, Ix, Iy of the I-beams and channels); dimensions and areas as reproduced on mechtool.cn, xycost.com and 164580.com; every row checked against the area from its dimensions (the standards' area formulas), misprints taken from the second source or corrected to the formula |
| `steel_aisc.csv` | W, HP, S, M, C, MC, L, HSS (rect / round), Pipe | AISC Shapes Database values as bundled in the `steelpy` package (Apache-2.0) |
| `steel_en.csv` | IPE, HE A / B / M, IPN, UPN, UPE, L | Section geometry from `structuralcodes` (fib-international, Apache-2.0); A and I computed with the fillets |
| `rebar.csv` | RB (TIS 20-2559), DB (TIS 24-2559), ASTM A615 #3-#18, JIS G3112 D6-D51, GB 1499 6-50 mm, EN 10080 6-50 mm | TISI full texts; ASTM A615 nominal dimensions; JIS G3112 nominal diameters (area 0.7854 d^2, 4 s.f.); EN 10080 nominal diameters |

Units: dimensions mm, A cm2, I and J cm4, mass kg/m (rebar: diameter mm, area mm2).

Notes
* Tapered flanges are drawn with the standard's slope (column `slope`, tan of the
  angle) and the tabulated thickness as the mean thickness at the middle of the
  flange: JIS / TIS channels 5 deg and I-beams 8 deg (from the area formulas of
  TIS 1227 annex A), GB channels 1:10 and I-beams 1:6, AISC C / S 1:6, EN UPN 8 % / 5 %,
  IPN 14 %.  The drawn areas then match the tables (TIS C150x75x6.5x10: 23.71 cm2,
  Iy 117 cm4 as tabulated).
* TIS 1228 / JIS G3350 shapes are drawn with the inside bend radius = t, which reproduces the tabulated areas
  exactly.  C125x50x20x4.0: Iy printed 33.1 in TIS 1228-2561; JIS G3350 (NSMP table) and the dimensions give
  31.1, which is used.
* TIS 1227 T-sections: the standard defines the stem only through its area (482.6 mm2 in the annex A formula,
  the same for every size); the stem is drawn with the uniform thickness of that area; A, Ix and Iy are the
  tabulated values.
* TIS 107 tubes: outside corner radius 2t (inside t), the standard's area formula (TIS 107-2533 annex A,
  JIS G3466).  The Pacific Pipe tables compute the square / rectangular tubes thicker than 6 mm with 2.5t
  corners; those JIS rows (112) are recomputed with 2t (they then agree with TIS 107-2533 and JIS G3466).
  EN 10219 tubes keep their corners (2t, 2.5t over 6 mm, 3t over 10 mm), ASTM A500 tubes the corners their
  table is computed with.  Left out: 15 ASTM A500 rows whose printed area, I and mass do not agree with
  each other (the AISC catalog has these sizes), BOX75x20x6 (2t corners do not fit).  TIS 107-2533 prints
  BOX100x50x4.0 Ix 142 / Iy 46.7; JIS G3466 and TIS 107-2566: 134 / 45.0 (used).  Five 2533 sizes missing
  from the 2566 tables (PIPE114.3x5.6, BOX350x350x9 / 12, BOX90x45x2.3 / 3.2) are taken from TIS 107-2533.
* The data is provided as is: check against the current standard or the
  supplier's table before design or purchase.

Apache-2.0 notices: steelpy (c) its authors; structuralcodes (c) fib - International
Federation for Structural Concrete.  Licence text: https://www.apache.org/licenses/LICENSE-2.0
