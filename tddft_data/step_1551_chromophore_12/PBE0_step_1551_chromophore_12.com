%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1551_chromophore_12 TDDFT with PBE1PBE functional

0 1
Mg   47.958   14.614   27.756
C   45.995   14.497   30.683
C   49.839   16.981   29.255
C   49.761   14.489   25.069
C   45.432   12.583   26.245
N   47.827   15.642   29.827
C   47.094   15.281   30.872
C   47.545   15.894   32.140
C   48.681   16.871   31.657
C   48.867   16.458   30.189
C   48.411   18.366   31.705
C   47.955   14.938   33.252
C   47.787   15.445   34.670
C   48.962   15.187   35.628
O   49.772   14.289   35.505
O   48.889   16.097   36.657
N   49.616   15.646   27.233
C   50.251   16.577   28.001
C   51.481   16.965   27.275
C   51.515   16.176   26.110
C   50.283   15.410   26.071
C   52.559   17.913   27.801
C   52.579   16.110   25.047
O   52.639   15.193   24.179
C   53.649   17.057   24.925
N   47.550   13.729   25.902
C   48.538   13.796   24.972
C   48.047   12.984   23.697
C   46.703   12.304   24.121
C   46.528   12.923   25.496
C   47.963   13.859   22.395
C   46.773   10.761   24.104
C   47.421   9.996   22.967
N   46.059   13.829   28.250
C   45.140   13.017   27.563
C   44.025   12.719   28.334
C   44.352   13.248   29.588
C   45.596   13.895   29.483
C   42.779   11.994   27.875
C   43.944   13.184   30.953
O   43.015   12.576   31.501
C   45.019   14.042   31.719
C   44.343   15.191   32.298
O   43.922   16.251   31.786
O   44.070   14.840   33.614
C   42.972   15.551   34.315
C   50.063   16.074   37.570
C   50.241   17.409   38.248
C   51.242   17.786   39.100
C   52.456   16.947   39.337
C   51.284   19.156   39.847
C   52.548   20.005   39.659
C   52.236   21.506   39.740
C   53.235   22.464   39.032
C   53.985   23.273   40.084
C   52.468   23.405   38.003
C   53.290   23.624   36.712
C   52.403   24.492   35.733
C   52.799   24.411   34.316
C   54.123   25.116   34.099
C   51.718   24.902   33.305
C   51.797   24.340   31.841
C   50.546   23.599   31.495
C   50.511   23.073   30.044
C   51.512   21.964   29.635
C   50.525   24.184   29.040
H   50.543   17.714   29.653
H   50.361   14.343   24.168
H   44.640   12.031   25.733
H   46.717   16.482   32.538
H   49.613   16.707   32.198
H   48.229   18.672   30.675
H   49.270   18.915   32.092
H   47.510   18.707   32.213
H   49.015   14.742   33.088
H   47.415   13.998   33.142
H   46.953   15.032   35.238
H   47.520   16.496   34.561
H   52.633   18.919   27.387
H   53.578   17.530   27.849
H   52.358   18.188   28.836
H   53.224   18.057   25.017
H   54.225   16.951   24.005
H   54.384   16.812   25.692
H   48.844   12.240   23.690
H   45.769   12.566   23.624
H   48.595   14.747   22.373
H   46.912   14.129   22.287
H   48.372   13.260   21.581
H   45.762   10.385   24.260
H   47.338   10.481   24.993
H   47.486   10.665   22.109
H   46.782   9.166   22.664
H   48.393   9.582   23.237
H   42.350   11.523   28.759
H   43.057   11.203   27.177
H   42.194   12.784   27.405
H   45.444   13.434   32.517
H   43.456   16.002   35.181
H   42.326   14.679   34.424
H   42.465   16.287   33.692
H   50.985   15.906   37.013
H   50.000   15.269   38.302
H   49.376   18.071   38.204
H   52.524   16.613   40.373
H   53.300   17.595   39.100
H   52.537   16.002   38.799
H   51.174   18.897   40.900
H   50.453   19.774   39.507
H   52.979   19.697   38.706
H   53.195   19.780   40.507
H   52.314   21.712   40.808
H   51.208   21.759   39.479
H   53.987   21.851   38.535
H   53.356   23.794   40.806
H   54.699   23.892   39.541
H   54.622   22.653   40.714
H   52.193   24.342   38.486
H   51.543   22.964   37.631
H   53.461   22.647   36.261
H   54.250   24.098   36.916
H   52.528   25.502   36.123
H   51.357   24.189   35.773
H   53.072   23.377   34.106
H   54.438   25.534   35.055
H   54.073   25.828   33.275
H   54.938   24.441   33.836
H   51.679   25.969   33.088
H   50.755   24.716   33.780
H   52.727   23.780   31.739
H   52.060   25.206   31.233
H   49.775   24.351   31.664
H   50.433   22.785   32.211
H   49.545   22.599   29.875
H   52.405   22.210   30.210
H   51.741   21.904   28.571
H   51.174   20.972   29.935
H   51.342   24.085   28.325
H   50.548   25.210   29.409
H   49.536   24.105   28.590

