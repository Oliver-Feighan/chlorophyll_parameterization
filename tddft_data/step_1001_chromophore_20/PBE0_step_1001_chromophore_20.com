%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1001_chromophore_20 TDDFT with PBE1PBE functional

0 1
Mg   6.606   57.271   41.575
C   5.141   54.050   41.466
C   9.512   55.796   40.493
C   7.647   60.392   40.667
C   3.464   58.655   42.263
N   7.221   55.150   40.951
C   6.463   54.063   41.042
C   7.257   52.845   40.533
C   8.710   53.372   40.697
C   8.521   54.860   40.738
C   9.481   52.834   41.934
C   6.974   52.615   39.012
C   7.020   51.092   38.551
C   6.085   50.615   37.500
O   5.055   49.921   37.654
O   6.371   51.232   36.249
N   8.352   57.978   40.717
C   9.504   57.212   40.426
C   10.495   58.065   39.934
C   9.946   59.412   39.861
C   8.549   59.280   40.350
C   11.898   57.628   39.644
C   10.690   60.613   39.458
O   11.860   60.504   39.161
C   10.069   61.965   39.233
N   5.799   59.311   41.575
C   6.394   60.408   41.175
C   5.583   61.653   41.298
C   4.209   61.067   41.776
C   4.453   59.567   41.827
C   6.278   62.660   42.259
C   2.914   61.500   41.056
C   1.711   61.726   41.864
N   4.639   56.495   41.957
C   3.517   57.215   42.206
C   2.464   56.211   42.396
C   3.086   54.943   42.107
C   4.431   55.214   41.799
C   0.956   56.413   42.661
C   2.898   53.540   42.179
O   1.932   52.910   42.539
C   4.218   52.861   41.701
C   3.958   52.078   40.551
O   3.621   52.420   39.436
O   4.046   50.782   40.923
C   3.293   49.796   40.142
C   5.400   50.930   35.214
C   5.820   51.582   33.979
C   5.454   51.267   32.695
C   4.344   50.149   32.273
C   6.206   52.052   31.583
C   5.922   53.552   31.768
C   5.293   54.270   30.557
C   5.514   55.818   30.636
C   4.051   56.383   30.771
C   6.423   56.445   29.470
C   7.772   57.000   29.975
C   8.715   56.010   30.710
C   9.817   55.406   29.852
C   10.252   54.037   30.398
C   11.051   56.298   29.617
C   11.391   56.434   28.157
C   12.766   57.167   27.992
C   13.945   56.267   27.754
C   15.211   56.875   28.307
C   14.180   55.812   26.233
H   10.477   55.306   40.350
H   7.972   61.361   40.283
H   2.483   59.076   42.492
H   7.059   51.947   41.117
H   9.403   53.207   39.872
H   10.422   52.502   41.496
H   8.923   52.043   42.435
H   9.594   53.699   42.587
H   7.829   53.071   38.514
H   6.062   53.116   38.686
H   6.970   50.425   39.411
H   8.038   50.892   38.217
H   12.693   58.204   40.118
H   12.068   57.749   38.574
H   12.134   56.601   39.920
H   9.765   62.436   40.168
H   9.169   61.878   38.625
H   10.717   62.664   38.704
H   5.497   62.069   40.294
H   4.091   61.312   42.832
H   5.486   63.136   42.837
H   6.713   63.428   41.618
H   6.873   62.134   43.005
H   2.681   60.700   40.354
H   3.056   62.389   40.442
H   1.852   61.428   42.903
H   0.922   61.091   41.460
H   1.329   62.741   41.753
H   0.602   57.183   41.976
H   0.881   56.786   43.682
H   0.450   55.455   42.533
H   4.699   52.279   42.487
H   3.821   48.866   40.353
H   3.121   50.124   39.117
H   2.315   49.659   40.604
H   4.439   51.380   35.462
H   5.330   49.888   34.903
H   6.650   52.257   34.186
H   3.877   49.837   33.207
H   4.944   49.432   31.713
H   3.708   50.661   31.551
H   5.902   51.826   30.561
H   7.282   51.995   31.752
H   6.784   54.112   32.131
H   5.196   53.600   32.579
H   4.241   54.020   30.416
H   5.754   53.883   29.648
H   6.020   56.180   31.531
H   4.147   57.461   30.639
H   3.541   56.239   31.723
H   3.542   55.964   29.903
H   5.907   57.298   29.029
H   6.529   55.788   28.607
H   7.655   57.953   30.490
H   8.303   57.295   29.070
H   8.237   55.167   31.210
H   9.329   56.607   31.385
H   9.246   55.183   28.950
H   9.410   53.344   30.365
H   10.587   53.997   31.434
H   11.069   53.704   29.758
H   11.918   55.820   30.071
H   10.910   57.281   30.065
H   10.593   56.940   27.614
H   11.449   55.431   27.735
H   12.986   57.781   28.866
H   12.579   57.868   27.179
H   13.789   55.317   28.264
H   15.674   56.296   29.106
H   15.084   57.940   28.500
H   16.018   56.969   27.580
H   15.138   56.184   25.870
H   13.412   56.238   25.587
H   14.147   54.730   26.105

