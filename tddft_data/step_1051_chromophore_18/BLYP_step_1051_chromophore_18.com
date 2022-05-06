%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1051_chromophore_18 TDDFT with blyp functional

0 1
Mg   35.196   49.133   25.463
C   35.074   47.246   28.524
C   33.761   51.615   27.274
C   34.443   50.530   22.595
C   36.092   46.205   23.843
N   34.191   49.303   27.669
C   34.484   48.460   28.681
C   33.996   49.089   30.048
C   33.424   50.491   29.569
C   33.885   50.550   28.088
C   31.912   50.744   29.844
C   35.258   49.233   30.927
C   34.948   50.061   32.171
C   33.642   49.872   32.904
O   33.100   48.799   33.020
O   33.132   51.067   33.381
N   34.285   50.811   24.958
C   33.813   51.791   25.827
C   33.467   52.979   25.098
C   33.659   52.682   23.737
C   34.156   51.299   23.714
C   33.104   54.384   25.829
C   33.497   53.585   22.473
O   33.757   53.282   21.296
C   33.092   55.048   22.606
N   35.377   48.440   23.487
C   35.014   49.269   22.490
C   35.270   48.662   21.110
C   36.032   47.302   21.484
C   35.828   47.292   23.006
C   34.057   48.467   20.017
C   37.582   47.266   21.096
C   38.674   47.359   22.143
N   35.674   47.127   26.026
C   36.009   46.057   25.233
C   36.236   44.876   25.993
C   36.012   45.313   27.354
C   35.600   46.649   27.287
C   36.451   43.503   25.399
C   36.058   44.925   28.725
O   36.449   43.869   29.217
C   35.463   46.099   29.551
C   34.354   45.546   30.301
O   33.276   45.222   29.832
O   34.675   45.453   31.645
C   33.675   44.785   32.478
C   31.904   50.898   34.121
C   31.454   52.289   34.542
C   31.295   52.840   35.722
C   31.587   52.359   37.184
C   30.808   54.273   35.794
C   31.872   55.322   35.530
C   31.302   56.574   34.866
C   30.778   57.673   35.819
C   31.941   58.410   36.541
C   29.845   58.689   35.225
C   28.512   58.836   35.971
C   28.753   59.722   37.188
C   28.078   61.042   37.076
C   29.113   62.161   37.405
C   26.723   61.136   37.873
C   25.953   62.407   37.520
C   25.832   63.373   38.740
C   25.397   64.817   38.396
C   26.512   65.892   38.450
C   24.085   65.144   39.102
H   33.410   52.470   27.856
H   34.308   51.001   21.620
H   36.575   45.343   23.377
H   33.345   48.347   30.510
H   33.968   51.301   30.053
H   31.486   49.782   30.130
H   31.444   51.043   28.906
H   31.892   51.505   30.623
H   36.130   49.685   30.455
H   35.420   48.217   31.286
H   35.093   51.096   31.863
H   35.721   49.823   32.902
H   33.222   54.377   26.913
H   32.071   54.702   25.687
H   33.775   55.196   25.549
H   33.622   55.599   23.383
H   32.043   54.974   22.889
H   33.088   55.621   21.678
H   36.022   49.296   20.639
H   35.495   46.448   21.071
H   33.080   48.706   20.438
H   33.993   47.423   19.710
H   34.249   49.007   19.090
H   37.816   48.018   20.342
H   37.811   46.372   20.517
H   38.196   47.678   23.069
H   39.479   48.036   21.857
H   39.123   46.366   22.147
H   37.020   43.585   24.473
H   35.454   43.070   25.316
H   37.049   42.970   26.138
H   36.269   46.539   30.139
H   33.305   43.844   32.071
H   32.785   45.387   32.663
H   34.102   44.670   33.475
H   31.918   50.226   34.979
H   31.101   50.593   33.450
H   31.277   52.981   33.719
H   30.665   52.166   37.732
H   32.341   52.903   37.753
H   32.067   51.385   37.080
H   30.461   54.559   36.787
H   29.878   54.351   35.232
H   32.694   55.012   34.886
H   32.168   55.657   36.524
H   30.421   56.214   34.335
H   32.070   56.881   34.156
H   30.333   57.187   36.687
H   32.886   58.062   36.125
H   31.895   58.262   37.620
H   32.007   59.485   36.373
H   29.490   58.290   34.274
H   30.284   59.653   34.969
H   28.246   57.873   36.407
H   27.702   59.248   35.369
H   29.810   59.794   37.446
H   28.394   59.327   38.138
H   27.822   61.214   36.031
H   29.563   62.322   36.426
H   29.849   61.937   38.178
H   28.580   63.082   37.640
H   26.803   61.117   38.960
H   26.177   60.210   37.690
H   25.030   62.181   36.985
H   26.575   62.869   36.753
H   26.874   63.349   39.057
H   25.125   62.856   39.388
H   25.078   64.782   37.354
H   26.728   66.045   37.393
H   27.369   65.472   38.977
H   26.252   66.834   38.932
H   24.208   65.964   39.810
H   23.768   64.395   39.828
H   23.302   65.364   38.376

