%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_651_chromophore_16 TDDFT with blyp functional

0 1
Mg   40.834   41.501   26.639
C   39.800   43.954   29.102
C   41.729   39.513   29.193
C   41.752   39.304   24.301
C   40.190   43.808   24.241
N   40.802   41.757   28.861
C   40.301   42.747   29.642
C   40.389   42.367   31.141
C   40.991   40.945   31.124
C   41.223   40.711   29.634
C   42.382   40.809   31.933
C   39.024   42.383   31.955
C   38.912   41.474   33.228
C   38.247   42.061   34.437
O   37.331   41.541   35.058
O   38.732   43.394   34.799
N   41.666   39.763   26.737
C   41.908   39.057   27.866
C   42.499   37.687   27.474
C   42.524   37.624   26.074
C   41.993   38.947   25.666
C   42.906   36.594   28.551
C   42.925   36.496   25.082
O   43.029   36.581   23.848
C   43.246   35.098   25.597
N   41.171   41.592   24.590
C   41.471   40.545   23.796
C   41.399   40.831   22.315
C   40.580   42.158   22.327
C   40.616   42.556   23.850
C   42.805   40.844   21.664
C   39.178   42.007   21.765
C   38.111   41.206   22.491
N   40.304   43.549   26.562
C   39.987   44.326   25.494
C   39.497   45.589   25.933
C   39.410   45.479   27.345
C   39.866   44.193   27.684
C   39.162   46.757   25.100
C   38.966   46.181   28.580
O   38.500   47.308   28.666
C   39.205   45.196   29.766
C   40.099   45.972   30.769
O   41.235   46.391   30.574
O   39.302   46.351   31.828
C   39.843   47.207   32.902
C   38.033   43.916   35.904
C   38.678   43.434   37.162
C   38.212   43.570   38.406
C   36.817   44.249   38.705
C   39.021   42.939   39.601
C   38.893   41.381   39.754
C   38.249   41.018   41.097
C   38.720   39.623   41.653
C   37.835   38.409   41.075
C   39.009   39.569   43.137
C   40.393   39.134   43.660
C   40.377   38.501   45.107
C   39.853   39.460   46.244
C   41.057   40.185   46.876
C   38.988   38.663   47.311
C   37.444   38.666   47.079
C   36.721   37.382   46.818
C   36.153   37.355   45.330
C   36.091   35.879   44.782
C   34.796   38.113   45.228
H   41.863   38.759   29.971
H   41.847   38.472   23.600
H   39.750   44.417   23.450
H   41.084   43.147   31.453
H   40.452   40.094   31.538
H   43.278   40.775   31.313
H   42.477   39.872   32.482
H   42.455   41.583   32.697
H   38.238   41.994   31.308
H   38.858   43.452   32.086
H   39.949   41.358   33.544
H   38.413   40.524   33.038
H   42.657   36.793   29.593
H   43.985   36.445   28.505
H   42.398   35.653   28.340
H   43.966   35.210   26.407
H   43.587   34.421   24.813
H   42.270   34.814   25.992
H   40.798   40.050   21.849
H   41.137   42.933   21.800
H   43.491   40.381   22.373
H   43.126   41.869   21.480
H   42.754   40.358   20.689
H   39.313   41.517   20.801
H   38.788   43.010   21.594
H   37.908   40.246   22.017
H   37.161   41.738   22.456
H   38.372   40.909   23.507
H   38.710   46.462   24.153
H   40.016   47.385   24.843
H   38.355   47.257   25.636
H   38.236   44.962   30.209
H   40.624   47.936   32.686
H   40.360   46.532   33.585
H   39.071   47.757   33.439
H   36.952   43.808   35.820
H   38.168   44.994   35.820
H   39.642   42.954   36.996
H   36.310   44.621   37.814
H   36.912   45.083   39.401
H   36.204   43.577   39.305
H   38.622   43.410   40.499
H   40.094   43.130   39.613
H   39.898   40.983   39.617
H   38.270   40.998   38.945
H   37.163   41.070   41.033
H   38.414   41.799   41.841
H   39.715   39.550   41.214
H   36.917   38.850   40.688
H   37.583   37.671   41.836
H   38.315   37.848   40.272
H   38.325   38.815   43.525
H   38.797   40.546   43.571
H   41.124   39.919   43.467
H   40.756   38.340   43.008
H   41.401   38.225   45.359
H   39.566   37.777   45.046
H   39.136   40.088   45.714
H   41.398   41.012   46.253
H   41.899   39.510   47.024
H   40.615   40.628   47.769
H   39.244   39.080   48.285
H   39.413   37.660   47.360
H   37.225   39.341   46.252
H   36.990   39.203   47.912
H   35.849   37.247   47.458
H   37.366   36.513   46.949
H   36.821   37.870   44.639
H   36.979   35.313   45.063
H   36.065   35.855   43.692
H   35.245   35.389   45.263
H   34.857   39.142   45.583
H   34.172   37.557   45.928
H   34.412   38.083   44.209

