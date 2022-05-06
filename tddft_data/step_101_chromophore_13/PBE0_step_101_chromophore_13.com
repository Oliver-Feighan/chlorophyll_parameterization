%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_101_chromophore_13 TDDFT with PBE1PBE functional

0 1
Mg   46.946   24.975   29.183
C   47.721   27.390   31.658
C   46.034   22.800   31.589
C   46.811   22.653   26.790
C   48.222   27.277   26.937
N   46.836   25.100   31.383
C   47.232   26.182   32.214
C   46.806   25.882   33.695
C   46.369   24.376   33.673
C   46.348   24.038   32.123
C   47.098   23.299   34.461
C   45.679   26.886   34.224
C   46.101   27.744   35.389
C   45.064   27.872   36.533
O   44.056   28.593   36.506
O   45.388   27.104   37.565
N   46.357   23.014   29.193
C   46.009   22.255   30.269
C   45.560   20.966   29.899
C   45.798   20.909   28.499
C   46.279   22.218   28.044
C   44.936   19.939   30.943
C   45.480   19.797   27.493
O   45.694   19.947   26.286
C   44.913   18.533   27.885
N   47.463   24.994   27.118
C   47.326   23.904   26.406
C   47.725   24.136   24.856
C   47.818   25.743   24.968
C   47.835   26.048   26.402
C   49.096   23.390   24.491
C   46.614   26.414   24.271
C   45.275   26.178   24.929
N   47.753   26.979   29.231
C   48.324   27.714   28.239
C   48.867   28.968   28.744
C   48.591   28.869   30.124
C   47.942   27.686   30.339
C   49.646   30.059   27.965
C   48.704   29.542   31.430
O   49.136   30.639   31.747
C   48.184   28.595   32.493
C   49.319   28.301   33.392
O   50.239   27.526   33.134
O   49.021   28.901   34.634
C   50.055   28.622   35.683
C   44.455   27.117   38.694
C   44.573   25.898   39.558
C   44.150   25.884   40.845
C   43.455   26.970   41.681
C   44.394   24.646   41.710
C   43.621   23.275   41.415
C   42.845   22.595   42.610
C   41.469   22.132   42.140
C   41.447   21.417   40.732
C   40.427   23.400   42.207
C   39.614   23.434   43.441
C   40.145   24.410   44.633
C   40.224   23.692   46.001
C   38.873   23.838   46.735
C   41.438   24.044   46.944
C   42.226   22.974   47.670
C   43.758   23.246   47.928
C   44.628   22.100   47.355
C   46.095   21.975   47.831
C   44.523   22.031   45.823
H   45.496   22.225   32.346
H   46.890   21.832   26.073
H   48.734   27.887   26.190
H   47.697   25.846   34.323
H   45.335   24.231   33.987
H   46.565   23.175   35.404
H   48.094   23.679   34.691
H   47.008   22.331   33.969
H   44.779   26.327   34.485
H   45.419   27.655   33.497
H   46.190   28.773   35.041
H   47.061   27.420   35.791
H   45.130   18.901   30.674
H   43.873   20.069   31.145
H   45.296   20.128   31.955
H   45.747   17.886   28.157
H   44.435   17.988   27.071
H   44.156   18.590   28.667
H   46.978   23.849   24.115
H   48.771   26.074   24.556
H   49.554   23.050   25.420
H   49.867   24.045   24.087
H   48.875   22.568   23.810
H   46.590   26.001   23.263
H   46.742   27.493   24.188
H   44.476   26.109   24.191
H   45.110   27.034   25.584
H   45.265   25.188   25.386
H   50.520   30.354   28.546
H   48.932   30.878   27.884
H   50.004   29.798   26.969
H   47.398   29.149   33.007
H   50.017   29.437   36.407
H   51.079   28.588   35.311
H   49.795   27.732   36.256
H   43.430   27.186   38.331
H   44.707   28.012   39.262
H   45.003   25.033   39.054
H   44.196   27.341   42.390
H   42.639   26.508   42.237
H   43.115   27.837   41.115
H   44.421   24.911   42.767
H   45.407   24.312   41.489
H   44.416   22.595   41.110
H   42.969   23.463   40.562
H   42.781   23.401   43.340
H   43.435   21.805   43.076
H   41.256   21.369   42.889
H   42.362   21.359   40.143
H   40.681   21.905   40.129
H   41.165   20.373   40.869
H   39.703   23.549   41.406
H   41.112   24.246   42.233
H   39.475   22.401   43.761
H   38.641   23.854   43.184
H   39.505   25.271   44.824
H   41.122   24.793   44.341
H   40.263   22.609   45.883
H   39.128   24.004   47.782
H   38.201   22.987   46.619
H   38.196   24.593   46.336
H   41.055   24.711   47.717
H   42.063   24.736   46.381
H   42.095   22.074   47.070
H   41.762   22.677   48.611
H   43.953   23.254   49.001
H   44.018   24.196   47.462
H   44.006   21.279   47.712
H   46.292   22.617   48.689
H   46.925   21.885   47.130
H   46.164   20.967   48.240
H   44.837   23.013   45.468
H   43.585   21.635   45.434
H   45.326   21.371   45.496

