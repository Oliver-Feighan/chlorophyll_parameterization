%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_301_chromophore_6 TDDFT with PBE1PBE functional

0 1
Mg   16.776   -2.360   27.938
C   16.023   -0.280   30.616
C   18.338   -4.524   30.044
C   17.872   -3.974   25.208
C   15.458   0.125   25.787
N   17.211   -2.347   30.081
C   16.812   -1.385   31.003
C   17.430   -1.735   32.451
C   18.260   -2.990   32.152
C   17.767   -3.375   30.663
C   19.803   -2.909   32.215
C   16.190   -1.971   33.499
C   16.610   -1.797   35.009
C   17.912   -1.051   35.376
O   18.342   -0.054   34.835
O   18.577   -1.674   36.421
N   17.850   -4.017   27.662
C   18.364   -4.813   28.723
C   19.067   -5.865   28.097
C   18.920   -5.843   26.717
C   18.195   -4.564   26.489
C   19.814   -6.905   28.841
C   19.366   -6.849   25.643
O   18.985   -6.661   24.455
C   20.190   -8.089   25.900
N   16.859   -1.903   25.813
C   17.311   -2.767   24.939
C   17.152   -2.255   23.526
C   16.201   -1.011   23.684
C   16.153   -0.894   25.174
C   18.520   -1.910   22.877
C   14.787   -1.237   23.020
C   14.765   -1.240   21.484
N   15.806   -0.516   28.062
C   15.245   0.319   27.151
C   14.466   1.368   27.835
C   14.782   1.208   29.244
C   15.659   0.081   29.272
C   13.835   2.586   27.309
C   14.578   1.710   30.562
O   13.891   2.667   30.924
C   15.387   0.743   31.555
C   16.410   1.642   32.215
O   17.477   1.939   31.748
O   15.993   1.954   33.438
C   16.725   2.982   34.140
C   19.891   -1.201   36.867
C   20.645   -2.292   37.479
C   20.617   -2.656   38.758
C   19.717   -2.168   39.830
C   21.420   -3.863   39.227
C   21.403   -5.102   38.343
C   21.009   -6.344   39.220
C   22.301   -7.343   39.443
C   21.847   -8.799   39.903
C   23.282   -6.808   40.456
C   24.616   -7.612   40.552
C   24.858   -8.123   41.961
C   25.654   -9.447   42.071
C   24.743   -10.679   42.059
C   26.427   -9.293   43.393
C   27.931   -8.983   42.996
C   28.804   -9.735   43.949
C   29.114   -11.209   43.617
C   30.019   -11.180   42.404
C   29.874   -11.854   44.803
H   18.703   -5.292   30.729
H   18.329   -4.415   24.320
H   14.814   0.792   25.209
H   18.156   -0.960   32.696
H   17.797   -3.739   32.795
H   20.087   -1.891   31.951
H   20.271   -3.600   31.514
H   20.109   -3.137   33.236
H   15.923   -3.028   33.494
H   15.261   -1.450   33.270
H   16.844   -2.763   35.458
H   15.750   -1.301   35.460
H   20.759   -7.052   28.318
H   19.276   -7.852   28.809
H   20.127   -6.498   29.803
H   19.714   -8.640   26.712
H   21.226   -7.779   26.041
H   19.989   -8.685   25.010
H   16.564   -2.822   22.805
H   16.668   -0.097   23.318
H   18.858   -0.894   23.083
H   18.541   -2.009   21.791
H   19.316   -2.549   23.258
H   14.176   -0.379   23.300
H   14.315   -2.134   23.422
H   15.748   -1.250   21.014
H   14.343   -0.276   21.200
H   14.121   -2.054   21.150
H   12.774   2.397   27.148
H   14.303   2.893   26.374
H   13.983   3.408   28.009
H   14.664   0.335   32.260
H   16.780   3.883   33.529
H   17.700   2.559   34.385
H   16.243   3.224   35.087
H   19.696   -0.283   37.421
H   20.389   -0.916   35.941
H   21.060   -2.904   36.678
H   19.089   -2.891   40.350
H   19.081   -1.339   39.521
H   20.355   -1.752   40.610
H   21.131   -3.995   40.270
H   22.459   -3.536   39.271
H   22.408   -5.242   37.946
H   20.721   -5.119   37.493
H   20.240   -6.875   38.659
H   20.514   -6.204   40.181
H   22.927   -7.365   38.551
H   20.788   -9.051   39.843
H   22.235   -8.911   40.915
H   22.438   -9.579   39.423
H   22.700   -6.721   41.373
H   23.533   -5.795   40.140
H   25.443   -6.983   40.224
H   24.579   -8.530   39.966
H   23.875   -8.296   42.399
H   25.431   -7.334   42.449
H   26.283   -9.604   41.195
H   25.290   -11.522   41.638
H   23.869   -10.408   41.466
H   24.540   -11.007   43.079
H   26.391   -10.139   44.079
H   26.052   -8.481   44.016
H   28.281   -7.957   43.104
H   28.271   -9.220   41.987
H   28.299   -9.855   44.908
H   29.732   -9.171   44.047
H   28.206   -11.809   43.547
H   29.508   -11.616   41.545
H   30.966   -11.709   42.516
H   30.413   -10.196   42.150
H   29.343   -12.805   44.836
H   29.739   -11.284   45.722
H   30.934   -12.037   44.626

