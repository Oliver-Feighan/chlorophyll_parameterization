%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1_chromophore_20 ZINDO

0 1
Mg   7.142   57.546   41.517
C   6.289   54.207   41.287
C   10.288   56.721   40.389
C   7.812   60.897   41.169
C   3.941   58.304   42.532
N   8.100   55.732   40.725
C   7.550   54.468   40.808
C   8.519   53.393   40.323
C   9.885   54.174   40.370
C   9.397   55.613   40.452
C   10.941   53.738   41.484
C   8.181   52.719   38.955
C   8.311   51.211   38.915
C   9.481   50.759   38.055
O   10.648   50.529   38.453
O   9.037   50.311   36.895
N   8.804   58.605   40.858
C   9.982   58.102   40.472
C   10.932   59.166   40.197
C   10.335   60.402   40.381
C   8.908   60.029   40.772
C   12.414   58.881   39.658
C   11.019   61.807   40.278
O   12.226   61.867   39.909
C   10.280   63.061   40.550
N   5.929   59.423   41.581
C   6.460   60.624   41.577
C   5.532   61.679   42.142
C   4.204   60.944   42.529
C   4.700   59.439   42.284
C   6.082   62.754   43.133
C   2.881   61.248   41.768
C   1.648   60.788   42.518
N   5.413   56.462   41.842
C   4.167   56.898   42.349
C   3.349   55.736   42.566
C   4.139   54.620   42.197
C   5.339   55.156   41.738
C   2.016   55.690   43.298
C   4.117   53.243   42.147
O   3.288   52.392   42.322
C   5.501   52.882   41.394
C   5.071   52.591   39.957
O   4.832   53.425   39.105
O   4.992   51.213   39.836
C   4.488   50.713   38.561
C   9.964   49.732   35.890
C   10.131   50.617   34.630
C   9.174   51.347   33.923
C   7.654   51.418   34.234
C   9.613   52.108   32.660
C   9.960   53.577   32.958
C   9.377   54.492   31.891
C   10.284   55.765   31.814
C   9.312   56.923   31.358
C   11.555   55.538   30.907
C   12.754   56.423   31.280
C   12.845   57.662   30.334
C   14.166   57.639   29.473
C   15.281   58.509   30.013
C   13.754   57.999   28.019
C   14.952   57.589   27.052
C   15.103   58.761   26.064
C   14.717   58.352   24.640
C   15.264   59.399   23.660
C   13.184   58.126   24.486
H   11.347   56.508   40.230
H   8.097   61.944   41.049
H   2.947   58.378   42.978
H   8.573   52.630   41.099
H   10.293   54.145   39.359
H   11.533   54.647   41.590
H   11.675   53.007   41.144
H   10.538   53.390   42.435
H   8.898   53.175   38.272
H   7.264   53.038   38.460
H   7.371   50.816   38.529
H   8.480   50.780   39.902
H   12.650   57.831   39.480
H   13.156   59.279   40.351
H   12.590   59.280   38.660
H   9.908   63.049   41.574
H   9.553   63.277   39.767
H   11.064   63.813   40.630
H   5.091   62.357   41.411
H   4.061   60.881   43.608
H   7.169   62.679   43.080
H   5.659   62.687   44.136
H   5.859   63.782   42.847
H   2.992   60.797   40.782
H   2.910   62.325   41.603
H   0.837   61.471   42.264
H   1.737   60.784   43.605
H   1.327   59.827   42.116
H   1.671   54.681   43.524
H   1.289   56.228   42.691
H   2.068   56.222   44.248
H   6.049   52.101   41.922
H   3.446   51.010   38.441
H   4.616   49.635   38.469
H   5.023   51.198   37.745
H   9.634   48.788   35.458
H   10.940   49.481   36.305
H   11.134   50.770   34.232
H   7.480   51.132   35.272
H   7.169   50.802   33.477
H   7.471   52.488   34.139
H   8.935   52.009   31.812
H   10.540   51.765   32.200
H   11.041   53.710   32.990
H   9.590   53.889   33.935
H   8.346   54.762   32.119
H   9.212   54.006   30.930
H   10.556   56.130   32.804
H   8.546   56.538   30.684
H   9.834   57.702   30.803
H   8.713   57.412   32.126
H   11.220   55.622   29.873
H   12.013   54.564   31.078
H   13.650   55.815   31.411
H   12.520   56.645   32.321
H   12.881   58.557   30.955
H   11.980   57.743   29.676
H   14.560   56.628   29.366
H   15.293   59.404   29.391
H   16.216   57.965   29.884
H   15.240   58.794   31.064
H   13.414   59.033   27.960
H   12.976   57.345   27.626
H   14.672   56.706   26.478
H   15.906   57.339   27.516
H   16.151   59.042   26.166
H   14.456   59.578   26.381
H   15.266   57.436   24.418
H   15.525   60.332   24.161
H   14.595   59.803   22.900
H   16.032   58.881   23.086
H   12.990   57.054   24.458
H   12.774   58.545   23.567
H   12.658   58.616   25.305

