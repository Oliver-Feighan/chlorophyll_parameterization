%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_751_chromophore_20 TDDFT with blyp functional

0 1
Mg   6.766   57.514   41.815
C   5.219   54.315   41.348
C   9.687   56.144   40.378
C   8.157   60.463   41.945
C   3.710   58.735   42.756
N   7.327   55.447   40.836
C   6.555   54.304   40.911
C   7.416   53.104   40.440
C   8.865   53.708   40.417
C   8.594   55.198   40.568
C   9.964   53.253   41.409
C   6.961   52.558   39.109
C   6.766   51.056   39.205
C   6.027   50.345   38.071
O   5.411   49.320   38.131
O   6.306   50.917   36.852
N   8.643   58.227   41.209
C   9.670   57.531   40.657
C   10.744   58.461   40.415
C   10.412   59.749   40.946
C   9.046   59.535   41.376
C   11.955   58.138   39.657
C   11.217   61.126   40.857
O   12.296   61.192   40.323
C   10.667   62.460   41.389
N   5.939   59.372   42.067
C   6.800   60.416   42.321
C   6.033   61.598   42.899
C   4.520   61.117   42.988
C   4.688   59.650   42.542
C   6.452   62.070   44.253
C   3.557   61.900   42.051
C   2.453   62.824   42.769
N   4.903   56.748   42.195
C   3.794   57.364   42.672
C   2.795   56.402   42.914
C   3.353   55.195   42.466
C   4.607   55.495   41.992
C   1.403   56.781   43.406
C   3.017   53.763   42.237
O   1.956   53.181   42.447
C   4.320   53.143   41.672
C   3.918   52.295   40.527
O   3.493   52.755   39.492
O   3.777   50.984   40.954
C   3.035   50.070   40.086
C   5.882   50.112   35.684
C   6.304   50.815   34.393
C   6.363   50.255   33.146
C   5.653   49.001   32.820
C   6.720   51.253   32.047
C   5.838   52.555   31.881
C   5.492   52.914   30.457
C   5.590   54.508   30.265
C   4.281   55.021   29.603
C   6.776   54.872   29.371
C   7.672   55.991   30.039
C   8.912   55.401   30.845
C   10.260   55.326   30.098
C   11.319   54.380   30.792
C   10.819   56.766   29.779
C   11.577   56.818   28.416
C   13.068   57.163   28.562
C   13.997   56.183   27.838
C   15.338   56.122   28.615
C   14.097   56.512   26.283
H   10.648   55.723   40.076
H   8.603   61.443   42.128
H   2.827   59.146   43.251
H   7.307   52.441   41.298
H   9.278   53.518   39.427
H   10.730   52.763   40.808
H   9.597   52.603   42.203
H   10.456   54.075   41.930
H   7.658   52.756   38.295
H   5.944   52.918   38.952
H   6.152   50.806   40.069
H   7.803   50.755   39.354
H   12.249   58.859   38.894
H   11.966   57.104   39.312
H   12.719   58.138   40.434
H   10.369   62.348   42.432
H   9.794   62.769   40.815
H   11.376   63.282   41.291
H   6.084   62.407   42.169
H   4.074   61.116   43.983
H   6.433   63.159   44.289
H   7.467   61.717   44.432
H   5.732   61.804   45.027
H   3.130   61.228   41.307
H   4.291   62.524   41.541
H   2.499   62.574   43.829
H   1.477   62.527   42.387
H   2.635   63.883   42.585
H   0.780   55.937   43.700
H   0.912   57.378   42.638
H   1.663   57.320   44.317
H   4.767   52.566   42.482
H   2.708   50.518   39.148
H   2.152   50.030   40.723
H   3.544   49.107   40.093
H   4.810   50.299   35.624
H   6.114   49.058   35.840
H   6.705   51.803   34.616
H   6.426   48.304   32.497
H   4.990   49.165   31.970
H   5.013   48.695   33.648
H   6.638   50.700   31.111
H   7.783   51.474   32.139
H   6.366   53.406   32.310
H   5.005   52.522   32.583
H   4.460   52.564   30.444
H   6.136   52.390   29.751
H   5.711   55.013   31.223
H   4.461   56.066   29.349
H   3.379   54.897   30.202
H   4.035   54.589   28.633
H   6.592   55.351   28.410
H   7.450   54.016   29.338
H   7.066   56.550   30.751
H   8.152   56.600   29.272
H   8.704   54.347   31.031
H   8.965   55.981   31.766
H   10.021   54.941   29.107
H   11.417   53.576   30.064
H   10.865   53.964   31.691
H   12.309   54.751   31.057
H   11.397   57.061   30.655
H   9.933   57.388   29.654
H   11.100   57.643   27.886
H   11.408   55.937   27.797
H   13.301   57.201   29.626
H   13.156   58.154   28.117
H   13.672   55.155   28.001
H   15.983   55.589   27.916
H   15.208   55.548   29.533
H   15.628   57.121   28.939
H   13.541   55.802   25.671
H   15.052   56.263   25.819
H   13.780   57.533   26.075
