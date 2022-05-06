%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1451_chromophore_6 TDDFT with blyp functional

0 1
Mg   17.193   -2.607   28.518
C   16.413   -0.355   31.195
C   19.043   -4.542   30.774
C   18.345   -4.564   25.916
C   16.182   -0.245   26.348
N   17.660   -2.477   30.724
C   17.184   -1.489   31.585
C   17.726   -1.658   32.969
C   18.810   -2.827   32.721
C   18.458   -3.308   31.312
C   20.349   -2.528   32.833
C   16.555   -1.949   34.020
C   16.794   -1.525   35.433
C   18.088   -0.790   35.798
O   18.315   0.407   35.545
O   18.954   -1.683   36.338
N   18.446   -4.359   28.414
C   19.077   -4.994   29.462
C   19.714   -6.167   28.971
C   19.557   -6.229   27.547
C   18.750   -4.996   27.234
C   20.352   -7.081   30.025
C   20.034   -7.211   26.442
O   19.753   -7.120   25.260
C   20.723   -8.514   26.854
N   17.134   -2.501   26.489
C   17.615   -3.379   25.623
C   17.387   -2.917   24.198
C   16.479   -1.685   24.323
C   16.645   -1.395   25.786
C   18.748   -2.641   23.462
C   15.007   -1.642   23.834
C   13.874   -2.128   24.774
N   16.279   -0.824   28.687
C   15.921   0.045   27.724
C   15.267   1.177   28.269
C   15.369   1.013   29.624
C   16.043   -0.208   29.813
C   14.546   2.304   27.427
C   15.156   1.752   30.836
O   14.575   2.822   31.023
C   15.821   0.877   31.937
C   16.755   1.680   32.806
O   17.969   1.789   32.819
O   15.932   2.329   33.667
C   16.566   2.948   34.861
C   20.277   -1.111   36.557
C   21.093   -2.307   37.074
C   21.943   -2.423   38.096
C   22.390   -1.254   39.058
C   22.574   -3.768   38.359
C   21.650   -4.864   38.936
C   22.025   -5.011   40.450
C   22.560   -6.340   40.722
C   21.508   -7.490   40.796
C   23.399   -6.186   42.004
C   24.652   -6.984   41.942
C   25.042   -7.674   43.323
C   24.925   -9.218   43.452
C   23.452   -9.582   43.724
C   25.847   -9.817   44.518
C   27.066   -10.418   43.854
C   28.072   -11.046   44.891
C   29.206   -10.117   44.991
C   30.006   -10.180   46.327
C   30.147   -10.358   43.748
H   19.425   -5.053   31.660
H   18.742   -5.045   25.020
H   15.810   0.508   25.649
H   18.347   -0.869   33.393
H   18.528   -3.684   33.332
H   20.440   -1.617   33.424
H   20.818   -2.203   31.904
H   20.770   -3.412   33.311
H   16.425   -3.024   33.889
H   15.656   -1.462   33.643
H   16.695   -2.505   35.899
H   16.064   -0.774   35.735
H   19.698   -7.866   30.403
H   20.828   -6.523   30.831
H   21.128   -7.555   29.424
H   20.269   -8.838   27.790
H   21.736   -8.243   27.151
H   20.749   -9.111   25.942
H   16.880   -3.818   23.853
H   16.957   -0.891   23.749
H   19.424   -3.495   23.512
H   19.267   -1.851   24.006
H   18.564   -2.407   22.413
H   14.855   -2.206   22.913
H   14.728   -0.604   23.650
H   13.075   -2.582   24.188
H   13.495   -1.421   25.512
H   14.309   -2.918   25.386
H   13.884   2.809   28.130
H   13.894   1.832   26.692
H   15.228   3.109   27.152
H   15.118   0.520   32.689
H   16.394   4.024   34.879
H   17.652   2.892   34.943
H   16.084   2.575   35.764
H   20.134   -0.366   37.340
H   20.767   -0.671   35.689
H   20.959   -3.175   36.427
H   23.465   -1.085   38.989
H   21.977   -1.345   40.063
H   21.904   -0.398   38.589
H   23.607   -3.704   38.701
H   22.799   -4.254   37.410
H   21.838   -5.748   38.328
H   20.609   -4.586   38.774
H   21.007   -4.936   40.833
H   22.553   -4.126   40.804
H   23.294   -6.460   39.924
H   20.640   -7.160   40.226
H   21.307   -7.671   41.852
H   21.989   -8.354   40.337
H   22.792   -6.541   42.837
H   23.718   -5.148   42.089
H   25.503   -6.340   41.724
H   24.641   -7.717   41.135
H   24.473   -7.242   44.146
H   26.086   -7.451   43.541
H   25.095   -9.659   42.470
H   23.087   -10.268   42.960
H   22.856   -8.678   43.850
H   23.437   -10.240   44.593
H   25.386   -10.605   45.113
H   26.203   -9.011   45.160
H   27.580   -9.649   43.277
H   26.733   -11.205   43.178
H   28.291   -12.061   44.559
H   27.529   -11.008   45.836
H   28.795   -9.107   44.975
H   29.685   -11.143   46.725
H   29.686   -9.328   46.926
H   31.087   -10.046   46.279
H   30.105   -9.658   42.914
H   30.027   -11.377   43.381
H   31.127   -10.162   44.183
