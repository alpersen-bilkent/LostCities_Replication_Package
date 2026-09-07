================================================================================
REPLICATION PACKAGE
Locating Lost Bronze Age Cities in Anatolia: A Comparative Study of
Multidimensional Scaling and Graph Convolutional Networks
================================================================================

This package reproduces BOTH of the paper's methods: the Graph
Convolutional Network (GCN, Sec. 2.2) and Multidimensional Scaling (MDS,
Sec. 2.1) -- leave-one-out validation on 15 known Bronze Age Anatolian
trading cities, and coordinate prediction for 10 cities whose locations
are lost, using the same ancient trade-itinerary data as the comparison
study (Barjamovic, Chaney, Cosar & Hortacsu, 2019), plus a modern-day
analogue using present-day inter-provincial trade in Turkey. Both methods
share the same underlying trade data, city coordinates, comparison
estimates, and plotting/screening code (src/data.py, src/config.py,
src/plots.py, src/terrain.py) -- only the model-fitting step itself
differs (src/train.py + src/model.py for GCN; src/mds.py for MDS).


--------------------------------------------------------------------------------
1. REQUIREMENTS
--------------------------------------------------------------------------------

Python 3.x (Anaconda distribution recommended).

Required packages:
    numpy, pandas, scikit-learn, scipy, tqdm, matplotlib, torch,
    torch_geometric, networkx

networkx is only used by experiments/plot_trade_network.py (Figure 1 --
the raw trade-itinerary network diagram, not a model result).

torch/torch_geometric are only actually used by GCN's own model code
(src/model.py, src/train.py) -- MDS's own fitting step (src/mds.py) is pure
numpy/scikit-learn -- but src/evaluate.py imports both GCN's and MDS's
driver functions from the same file, so importing it at all (which every
experiments/mds_*.py script does) currently pulls in torch/torch_geometric
regardless. Splitting evaluate.py so MDS-only runs wouldn't need them was
considered and deliberately not done, to keep one shared driver file rather
than two -- not expected to matter in practice since this environment
already needs both packages for the GCN scripts anyway.

Optional packages (needed only for the geographic screening step and the
topographic basemap layer on the maps, for both methods):
    geopandas, contextily, rasterio, shapely

All other results (model training, leave-one-out validation, lost-city
coordinate prediction) run with only the required packages above.


--------------------------------------------------------------------------------
2. DATA SOURCES
--------------------------------------------------------------------------------

Ancient trade data (cuneiform tablet itineraries, 15 known and 10 lost
city names and known coordinates):
    Barjamovic, G., Chaney, T., Cosar, K., & Hortacsu, A. (2019).
    "Trade, Merchants, and the Lost Cities of the Bronze Age."
    Quarterly Journal of Economics, 134(3), 1455-1503.
    https://doi.org/10.1093/qje/qjz009
    Data is drawn from the authors' own public replication package for
    that paper. This package embeds the trade matrix and known-city
    coordinates directly (src/data.py) rather than requiring a separate
    download.

Modern-day inter-provincial trade data (Iller Arasi Ticaret.xlsx):
    T.C. Sanayi ve Teknoloji Bakanligi. Girisimci Bilgi Sistemi (GBS)
    Iller Arasi Ticaret Raporu. Republic of Turkiye Ministry of Industry
    and Technology (2024).
    https://gbs.sanayi.gov.tr/Home/Report
    Accessed: Dec. 15, 2025.

Turkey administrative boundary outline (used to draw country borders on
the maps):
    GADM, https://gadm.org

Elevation, river/lake geometry, and known-archaeological-site and
mineral-deposit reference data used in the geographic screening step
(Section 7 below):
    From the same Barjamovic, Chaney, Cosar & Hortacsu (2019) replication
    package cited above. Known-site coordinates are drawn from that
    paper's own supplemental appendix (its table of candidate
    archaeological sites for lost cities); mineral deposit locations are
    from that paper's own compiled mining-district data (originally
    De Jesus, 1980; Massa, 2016).

All of the above -- the modern trade spreadsheet, the GADM shapefile, and
the specific Barjamovic et al. replication-package files this codebase
reads -- are bundled directly in third_party_data/, so this package runs
end to end with no separate download. See Section 8 for exactly what's
included, and its provenance/licensing.


--------------------------------------------------------------------------------
3. REPOSITORY STRUCTURE
--------------------------------------------------------------------------------

    experiments/         One script per reported result. RUN THESE.
    src/                 Shared model, training, and plotting code,
                          imported by every script in experiments/. Not
                          an entry point; do not run these files directly.
    legacy/               Earlier exploratory versions of the analysis
                          (both GCN's and MDS's), kept only for
                          transparency. Superseded; do not run.
    results/             Output of every run -- tables (.parquet/.csv)
                          and figures (.png). Regenerated automatically
                          by rerunning experiments/; safe to delete.
    third_party_data/    Bundled third-party inputs (modern trade data,
                          Turkey shapefile, the specific Barjamovic et
                          al. replication-package files this code reads)
                          -- see Section 8. Not generated by this code;
                          do not delete.
    tests/               Verification scripts written during this
                          revision to check specific fixes against
                          synthetic ground truth (e.g. the dyadic-share
                          formula, the significance test). Not part of
                          the reported pipeline; run standalone if you
                          want to see that check for yourself.
    CHANGELOG_AND_HANDOFF.md
                          What changed vs. the original scripts, why,
                          and how each fix was verified -- covers GCN,
                          then MDS, in that order.
    MANUSCRIPT_REVISION_GUIDE.md
                          Point-by-point map from each referee comment to
                          what's already been fixed/built, what still
                          needs new manuscript text, and what needs an
                          editorial decision -- includes the current,
                          authoritative headline numbers (from
                          results/Tables.xlsx) and every new citation
                          this revision requires. Read this one first if
                          you're picking up the manuscript rewrite itself
                          rather than the code.


--------------------------------------------------------------------------------
4. HOW TO RUN
--------------------------------------------------------------------------------

Run any file in experiments/ directly (e.g. from Spyder, or from a
terminal as "python experiments/<file>.py"). Each script is
self-contained and saves its own results. MDS scripts are prefixed
"mds_"; GCN scripts have no prefix.

--- Figure 1 (either before or independent of GCN/MDS below) ---

    experiments/plot_trade_network.py

Not a model result -- just the raw, symmetrized trade-itinerary network
(all 25 cities, known in blue, lost in orange), the same figure the
paper shows before introducing either method. Draws directly on
src/data.py, nothing else in this package depends on it.

--- Known-cities overview map (Appendix A) ---

    experiments/plot_known_cities.py

Also not a model result -- an overview map of the 15 known cities on
the Anatolian basemap (Figures/KnownCitiesTurkey.png in the paper's
Appendix A), for orientation before either method's predictions are
shown. Draws city names and coordinates from the same
data.ancient_known_coords() lookup every experiment script uses, so it
cannot drift out of sync with the actual known-city coordinates the
way an earlier, unrecovered ad-hoc version of this figure once did.

--- GCN ---

Step 1 -- Hyperparameter search (run first)

    experiments/grid_search_ancient_dyadic.py
    experiments/grid_search_ancient_tii.py
    experiments/grid_search_modern_dyadic.py
    experiments/grid_search_modern_tii.py

Each of these prints a chosen (learning rate, embedding size, epoch
count) and saves a search figure. The result is already set as the
default in the matching script below, so this step does not need to be
repeated to reproduce the paper's numbers -- rerun it only if the
underlying trade data changes.

Step 2 -- Main results (any order)

    experiments/modern_loo_dyadic.py
    experiments/modern_loo_tii.py
    experiments/ancient_loo_dyadic.py
    experiments/ancient_loo_tii.py
    experiments/ancient_lost_dyadic.py
    experiments/ancient_lost_dyadic_constrained.py

Step 3 -- Geographic screening (run after ancient_lost_dyadic_constrained.py in Step 2)

    experiments/screening_ancient_lost.py

--- MDS ---

No hyperparameter search step: MDS has no learning-rate/epoch/embedding-
dimension to tune the way GCN does (SMACOF + weighted Procrustes is
deterministic given a dissimilarity matrix and a random seed).

Step 1 -- Main results (any order)

    experiments/mds_modern_loo_dyadic.py
    experiments/mds_modern_loo_tii.py
    experiments/mds_ancient_loo_dyadic.py
    experiments/mds_ancient_loo_tii.py
    experiments/mds_ancient_lost_dyadic.py

No mds_ancient_lost_tii.py / no "_constrained" variant: dyadic share
outperformed TII in MDS's own LOO training, so (consistent with the
original scripts) only dyadic is used for the actual lost-city
prediction; directional-constraint training (Sec. 7.1, GCN only) doesn't
have a natural equivalent in MDS's closed-form Procrustes alignment, so
was not attempted for MDS -- see CHANGELOG_AND_HANDOFF.md's MDS section.

Step 2 -- Geographic screening (run after mds_ancient_lost_dyadic.py in Step 1)

    experiments/mds_screening_ancient_lost.py


--------------------------------------------------------------------------------
5. WHAT EACH SCRIPT PRODUCES
--------------------------------------------------------------------------------

plot_trade_network.py
    Figure 1: the raw, symmetrized trade-itinerary network for all 25
    cities on a ring, known cities in blue, lost in orange, edge
    width/label showing itinerary counts. No model, no prediction --
    describes the input data only.

plot_known_cities.py
    Appendix A overview figure: the 15 known ancient cities plotted on
    the Anatolian basemap with name labels, cropped to the same
    31-39.2 E / 35.8-42.25 N window used throughout the paper's other
    maps. No model, no prediction -- describes the input data only.

modern_loo_dyadic.py / modern_loo_tii.py
    Predicted coordinates, standard deviation, and error vs. actual
    location, for each of the 15 modern Turkish province centers,
    leave-one-out. One uncertainty map per city plus one combined map.

ancient_loo_dyadic.py / ancient_loo_tii.py
    Same, for the 15 known ancient cities, fictitiously held out one at
    a time and re-predicted from the rest of the trade network. Also
    runs a significance test against the comparison study's own headline
    error -- see Section 6 below.

ancient_lost_dyadic_constrained.py
    The reported lost-city prediction. Predicted coordinates, standard
    deviation, and distance to three independent comparison estimates,
    for the 10 actually-lost cities. The model is additionally penalized
    during training for violating directional relationships documented
    in ancient texts (for example, "City X lies south and east of
    Kanesh"), compiled by historians Forlanini (2008) and Barjamovic
    (2011). Per-city maps show the statistical confidence region, the
    geography-screened candidate area, and any known archaeological site
    or mineral deposit inside it, plus one combined map of all 10.

ancient_lost_dyadic.py
    The same prediction task without the directional-constraint penalty.
    Kept as a robustness/ablation check showing what trade topology
    alone predicts, with no outside textual evidence injected -- that
    predictive-power claim is already established by the LOO and
    modern-city results above, so it does not need to be re-proven here.

screening_ancient_lost.py
    A finer-grained ranking of individual points inside each city's
    confidence region, by slope, distance to water, distance to the
    nearest known archaeological site, and distance to the model's own
    predicted coordinates (from ancient_lost_dyadic_constrained.py's
    output by default). Any known archaeological site inside the region
    is folded into this same ranked pool and a plot, so it's immediately
    visible whether a real, named site happens to sit in a well- or
    poorly-ranked spot.

mds_modern_loo_dyadic.py / mds_modern_loo_tii.py
mds_ancient_loo_dyadic.py / mds_ancient_loo_tii.py
mds_ancient_lost_dyadic.py
    MDS counterparts of the five GCN scripts immediately above --
    identical task, identical reported quantities, same trade data and
    comparison estimates, only the model-fitting step differs (SMACOF
    embedding + weighted Procrustes, src/mds.py, instead of a GCN).
    mds_ancient_loo_dyadic.py/mds_ancient_loo_tii.py also run the same
    significance test as their GCN counterparts (Sec. 6 below).

mds_screening_ancient_lost.py
    MDS counterpart of screening_ancient_lost.py below, reading
    mds_ancient_lost_dyadic.py's output instead of GCN's. Writes its
    per-city CSVs as results/screening/mds_{city}.csv (the "mds_" prefix
    avoids overwriting GCN's results/screening/{city}.csv for the same
    city name).

Every leave-one-out and lost-city table (GCN or MDS) reports the same
four quantities per city: mean predicted longitude and latitude, their
bootstrap standard deviation, and an error or distance column -- so all
result tables are read the same way. Tables print to the console and
save to results/*.parquet regardless of how the script is run.


--------------------------------------------------------------------------------
6. STATISTICAL SIGNIFICANCE OF THE HEADLINE COMPARISON
--------------------------------------------------------------------------------

ancient_loo_dyadic.py and ancient_loo_tii.py (GCN), and
mds_ancient_loo_dyadic.py and mds_ancient_loo_tii.py (MDS), each also test
whether their leave-one-out result is significantly better than the
comparison study's own headline out-of-sample error (116.03 km), not just
numerically lower. The test itself (src/evaluate.py's
bootstrap_ensemble_mean_distribution() / significance_test_vs_reference())
is entirely method-agnostic -- it operates on the B replicate predictions
already produced by either method's LOO run, so the same function and the
same statistical reasoning apply unchanged to both.

An earlier version of this section claimed that comparison figure is
published as a single aggregate number with no city-by-city breakdown,
so a paired test (e.g. Wilcoxon signed-rank, matching city to city) was
not something the published data supported. That was wrong -- Barjamovic
et al.'s own per-city known-city estimates ARE available (their Appendix
Table 2 robustness exercise; already embedded in this codebase as
data.BARJAMOVIC_KNOWN_ESTIMATES, spot-verified to reproduce their own
printed per-city errors, e.g. Hattus ~=133.3 km, to within rounding --
confirming 116.03 really is the mean of a genuine per-city table). Each
of the four scripts above now ALSO runs a second, paired test alongside
the one-sample test described below -- see "THE PAIRED, CITY-MATCHED
TEST" further down this section. Both are kept and reported side by
side; the paired version doesn't replace the aggregate one, since they
answer different questions (unpaired aggregate-vs-aggregate here; a
like-for-like, city-matched comparison there).

The reported result for each city is the AGGREGATED prediction: the
mean of that city's 200 independently Poisson-resampled and retrained
predictions, scored against the true location -- this is what
summarize_bootstrap() reports as Error_KM and what the console prints as
"Overall Mean Error." Averaging predictions before scoring them, rather
than averaging their individual scores, is a form of bootstrap
aggregation ("bagging" -- Breiman, 1996, "Bagging Predictors," Machine
Learning 24(2):123-140), and its whole purpose is variance reduction: it
partly cancels out each replicate's independent noise, so a
single-replicate error and the aggregated prediction's error are
different quantities, not interchangeable. The significance test has to
target the one actually reported.

The standard nonparametric approach to estimating the sampling
variability of a statistic computed from resampled data -- here, "the
error of a size-200 average of replicate predictions" -- is to resample,
with replacement, from the replicates already on hand, and recompute the
statistic on each resample (Efron, 1979, "Bootstrap Methods: Another
Look at the Jackknife," Annals of Statistics 7(1):1-26; Efron &
Tibshirani, 1993, An Introduction to the Bootstrap, Chapman & Hall/CRC,
chs. 6 and 13 for percentile confidence intervals). No retraining is
needed for this: the 200 replicate predictions each script already
produces are exactly the empirical sample this resamples from, 2,000
times by default, to build a distribution of "what the aggregated
predictor's overall error would look like" under resampling.

One detail that matters for getting an honest interval width: all 15
cities' predictions for a given replicate came from the same
Poisson-perturbed matrix and the same training run, so a replicate that
happens to train unusually well or badly can move every city's error in
the same direction at once. The resampling draws one shared set of
replicate indices per resample and reuses it across every city, so that
correlation is preserved rather than discarded -- resampling each city
independently would understate the interval's width by treating cities
as independent when they are not.

The script reports the mean and 95 percent interval of that resampled
distribution, and the fraction of resamples that did NOT beat the
116.03 km reference figure. A small fraction is direct evidence that the
result holds up under resampling, not that it depends on exactly how the
200 replicates happened to average out.

Printed to the console under "SIGNIFICANCE TEST" and saved to
results/ancient_loo_dyadic_significance.csv,
results/ancient_loo_tii_significance.csv (GCN), and
results/mds_ancient_loo_dyadic_significance.csv,
results/mds_ancient_loo_tii_significance.csv (MDS).

THE PAIRED, CITY-MATCHED TEST

Each of the four scripts above also runs a second test
(src/evaluate.py's paired_significance_test_vs_reference()), comparing
this method's error against Barjamovic et al.'s own error for the SAME
city, city by city, instead of only aggregate-vs-aggregate. This is the
more appropriate comparison in one respect the one-sample test above
cannot capture: some known cities are inherently harder to locate for
BOTH methods (sparser trade records), and a paired test controls for
that by only ever comparing a city against itself, never against a
different city's difficulty.

A classical Wilcoxon signed-rank test (scipy.stats.wilcoxon, one-sided,
EXACT method -- not the large-sample normal approximation, which n=15
is too small to justify) is run on the point estimates: this method's
own per-city error (the distance from the bootstrap-mean prediction --
the "middle point" of the B replicates -- to the true coordinate, i.e.
exactly what summarize_bootstrap() reports as Error_KM per city) vs.
Barjamovic's same per-city error, 15 matched pairs. This is the standard
test for comparing two methods' errors across the same N paired test
cases -- see Demsar (2006), "Statistical Comparisons of Classifiers over
Multiple Data Sets," Journal of Machine Learning Research 7, 1-30, the
standard reference for using Wilcoxon signed-rank this way.

n=15 known cities is a modest sample -- this affects the test's POWER
(its ability to detect a real but small effect), not its validity. The
result is a paired comparison specific to this exact, complete set of
15 known Bronze Age cities with recoverable coordinates, not a claim
about how the two methods would compare on some different or larger set
of cities.

Printed to the console under "PAIRED (CITY-MATCHED) WILCOXON TEST" and
saved to results/ancient_loo_dyadic_paired_significance.csv,
results/ancient_loo_tii_paired_significance.csv (GCN), and
results/mds_ancient_loo_dyadic_paired_significance.csv,
results/mds_ancient_loo_tii_paired_significance.csv (MDS).

If the four LOO scripts above were already run BEFORE this test existed,
there's no need to rerun them just for this -- their results/*.parquet
files already have everything this test needs (each city's Error_KM,
unchanged by this addition). Run experiments/recompute_paired_tests.py
instead: it reads those existing files directly and only computes this
one cheap test, no retraining or bootstrapping, done in seconds.


--------------------------------------------------------------------------------
7. COMPARISON ESTIMATES AND THE GEOGRAPHY-SCREENED REGION
--------------------------------------------------------------------------------

This section describes both GCN's and MDS's lost-city output identically
-- src/terrain.py takes any (mean_long, mean_lat, std_long, std_lat)
regardless of which method produced it, so mds_ancient_lost_dyadic.py and
mds_screening_ancient_lost.py apply everything below unchanged, just
against MDS's own predicted ellipses instead of GCN's.

For the 10 lost cities, three independent comparison estimates are
reported, and should not be confused with one another:

    Dist_to_Baj              Barjamovic et al.'s own fitted structural
                              gravity-model estimate (their paper's
                              primary result).
    Dist_to_Barjamovic2011   Barjamovic's own separate 2011
                              historical-geography proposal -- expert
                              philological judgment, not a fitted model.
    Dist_to_Forlanini2008    Forlanini's 2008 historical-geography
                              proposal, likewise expert judgment.

The latter two also appear as "B" and "F" markers on the per-city maps
only, omitted from the combined map to avoid clutter.

The geography-screened region (the shaded outline on ancient_lost_
dyadic_constrained.py's per-city maps) is not a weighted or scored estimate. It
starts from the model's statistical confidence region and removes areas
that fail a hard, independently justifiable test -- slope over 15
degrees, or inside a river channel or lake -- with no blending of
criteria and no fitted weights. It currently does not screen on
aridity or other soil/climate suitability; it may still include large,
mostly-passable areas for some cities as a result.

Any known archaeological site or mineral deposit that falls inside this
region is listed in a companion table (results/known_sites/ and
results/minerals/) and marked on the per-city map. Mineral deposits are
shown for context only -- they are never used to include, exclude, or
rank a candidate area, because Barjamovic et al.'s own analysis found
distance to known copper deposits not to be a significant or robust
predictor of ancient city location.


--------------------------------------------------------------------------------
8. BUNDLED THIRD-PARTY DATA
--------------------------------------------------------------------------------

third_party_data/ contains everything this codebase reads from an
external source, so the whole pipeline -- model training, leave-one-out
validation, lost-city coordinate prediction, the geographic screening
step, and the topographic basemap layer on every map -- runs end to end
on a fresh machine with nothing else to download or configure.

    third_party_data/modern_trade_data/
        Iller Arasi Ticaret.xlsx -- T.C. Sanayi ve Teknoloji Bakanligi
        (2024), cited in full in Section 2 above.

    third_party_data/turkey_shapefile/
        gadm41_TUR_0/1/2.* -- GADM's Turkey administrative boundaries
        (country/province/district level; only level 1, province, is
        actually used by src/plots.py). https://gadm.org

    third_party_data/barjamovic_replication_package/
        Only the specific files this codebase reads from Barjamovic,
        Chaney, Cosar & Hortacsu's (2019) own public replication
        package -- a few hundred KB, not their full package (~2GB,
        mostly other papers' figures and unused raw data this pipeline
        never touches):
          figures_tables/GEOdata/fao_gaez_elevation/faodata-clipped.tif
              -- FAO-GAEZ elevation raster, clipped to Turkey.
          figures_tables/GEOdata/ancient_mineral_deposits/ancientminedata.dta
              -- mineral deposit locations (De Jesus 1980; Massa 2016).
          roadknots/Input/rivers/*.shp
              -- river/lake geometry (Natural Earth + a Turkey lakes
              layer), Barjamovic et al.'s own copy.
          roadknots/Input/crossing_coord.xlsx
              -- river-crossing coordinates, loaded for completeness;
              not currently used by the screening ranking itself (see
              src/terrain.py's module docstring).
        The full replication package is publicly available via the
        paper's own citation (Section 2 above) if you need anything
        beyond these specific files.

Every path in src/config.py that points into third_party_data/ can be
redirected with an environment variable instead (GCN_MODERN_TRADE_FILE,
GCN_SHAPEFILE_PATH, GCN_REPLICATION_PACKAGE_DIR) -- useful if you'd
rather point at your own copy, or a newer version of any of the above,
without editing source.
