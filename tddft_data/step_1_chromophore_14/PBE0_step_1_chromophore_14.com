%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1_chromophore_14 TDDFT with PBE1PBE functional

0 1
Mg   46.693   44.583   44.387
C   43.255   44.724   43.603
C   46.826   41.397   42.973
C   50.080   44.926   44.229
C   46.416   48.036   45.298
N   45.217   43.253   43.458
C   43.915   43.522   43.224
C   43.244   42.376   42.399
C   44.333   41.221   42.580
C   45.596   41.991   43.085
C   44.097   40.057   43.569
C   43.052   42.863   40.913
C   41.752   42.343   40.239
C   41.896   41.904   38.725
O   42.035   40.757   38.333
O   41.726   42.978   37.892
N   48.302   43.231   43.873
C   48.096   41.975   43.367
C   49.372   41.386   43.215
C   50.371   42.366   43.492
C   49.618   43.584   43.943
C   49.623   39.975   42.738
C   51.834   42.251   43.230
O   52.315   41.279   42.670
C   52.807   43.268   43.533
N   48.039   46.332   44.626
C   49.424   46.113   44.664
C   50.137   47.336   45.150
C   48.993   48.377   45.439
C   47.744   47.576   45.114
C   50.982   46.967   46.376
C   49.231   49.783   44.770
C   49.254   50.972   45.682
N   45.147   46.130   44.494
C   45.211   47.430   44.932
C   43.854   48.013   45.018
C   43.095   46.966   44.431
C   43.894   45.857   44.225
C   43.427   49.418   45.413
C   41.786   46.667   43.914
O   40.774   47.382   43.847
C   41.780   45.236   43.377
C   40.729   44.427   44.022
O   39.867   43.906   43.412
O   40.962   44.180   45.344
C   40.024   43.159   45.859
C   41.777   42.730   36.408
C   43.154   43.148   35.869
C   44.189   42.476   35.324
C   44.396   40.979   35.259
C   45.435   43.175   34.810
C   45.594   43.174   33.187
C   45.257   44.505   32.513
C   45.455   44.397   30.956
C   45.844   45.820   30.468
C   44.261   43.879   30.205
C   44.580   42.848   29.110
C   43.908   43.294   27.771
C   45.002   43.772   26.724
C   44.535   45.099   26.116
C   45.384   42.712   25.647
C   46.944   42.991   25.235
C   47.051   43.070   23.695
C   47.458   41.714   23.016
C   48.963   41.548   23.150
C   46.873   41.676   21.522
H   46.890   40.369   42.609
H   51.171   44.934   44.244
H   46.143   49.074   45.502
H   42.284   42.145   42.861
H   44.788   40.862   41.657
H   43.419   40.367   44.365
H   44.986   39.616   44.020
H   43.583   39.286   42.996
H   43.924   42.613   40.308
H   43.016   43.950   40.986
H   41.070   43.189   40.326
H   41.426   41.560   40.924
H   50.584   39.537   43.008
H   49.534   40.056   41.655
H   48.816   39.258   42.889
H   52.778   43.418   44.613
H   52.610   44.182   42.972
H   53.790   42.904   43.234
H   50.707   47.794   44.342
H   48.710   48.541   46.479
H   51.115   45.892   46.500
H   50.522   47.312   47.302
H   52.019   47.252   46.201
H   48.426   49.876   44.041
H   50.214   49.895   44.314
H   48.516   51.679   45.304
H   50.215   51.486   45.655
H   49.143   50.709   46.734
H   43.515   50.131   44.593
H   43.932   49.786   46.307
H   42.356   49.273   45.554
H   41.466   45.238   42.334
H   40.254   43.001   46.913
H   40.121   42.229   45.299
H   38.985   43.484   45.901
H   41.105   43.531   36.101
H   41.387   41.797   36.000
H   43.170   44.238   35.847
H   44.305   40.669   34.218
H   43.700   40.374   35.840
H   45.378   40.564   35.485
H   46.348   42.701   35.171
H   45.392   44.221   35.111
H   44.977   42.350   32.830
H   46.588   42.814   32.922
H   45.977   45.156   33.010
H   44.267   44.908   32.724
H   46.337   43.804   30.713
H   46.470   46.340   31.193
H   44.912   46.378   30.375
H   46.392   45.655   29.540
H   43.646   44.722   29.890
H   43.551   43.429   30.898
H   44.150   41.886   29.389
H   45.648   42.680   28.968
H   43.201   44.089   28.007
H   43.374   42.455   27.324
H   45.823   43.950   27.419
H   45.026   45.315   25.167
H   44.887   45.786   26.885
H   43.452   45.123   25.998
H   44.629   42.577   24.873
H   45.515   41.743   26.129
H   47.565   42.178   25.612
H   47.338   43.917   25.653
H   47.678   43.882   23.327
H   46.040   43.245   23.326
H   46.940   40.928   23.567
H   49.307   42.353   23.800
H   49.433   41.543   22.167
H   49.187   40.623   23.680
H   46.471   42.638   21.206
H   46.239   40.822   21.284
H   47.748   41.617   20.875

