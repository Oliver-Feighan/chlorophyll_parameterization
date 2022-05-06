%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1_chromophore_2 ZINDO

0 1
Mg   3.158   -0.094   44.312
C   5.981   1.967   44.127
C   1.598   2.251   42.305
C   0.535   -2.151   44.137
C   5.149   -2.670   45.585
N   3.887   1.897   43.132
C   5.038   2.565   43.339
C   5.131   3.861   42.570
C   3.590   4.004   42.271
C   2.946   2.644   42.520
C   2.955   5.033   43.244
C   6.136   3.755   41.416
C   6.210   2.461   40.579
C   6.195   2.644   39.066
O   6.007   3.683   38.400
O   6.519   1.434   38.422
N   1.341   0.027   43.380
C   0.866   1.125   42.713
C   -0.598   1.037   42.635
C   -0.951   -0.318   43.020
C   0.339   -0.855   43.551
C   -1.479   2.108   41.885
C   -2.325   -0.981   42.895
O   -3.285   -0.340   42.569
C   -2.490   -2.436   43.227
N   2.984   -2.239   44.619
C   1.732   -2.881   44.593
C   1.803   -4.396   45.061
C   3.316   -4.491   45.581
C   3.860   -3.019   45.284
C   0.723   -4.977   46.104
C   4.234   -5.645   44.969
C   5.280   -6.178   46.068
N   5.163   -0.356   44.840
C   5.798   -1.431   45.428
C   7.171   -1.059   45.747
C   7.330   0.289   45.334
C   6.084   0.653   44.707
C   8.162   -1.875   46.535
C   8.239   1.467   45.150
O   9.427   1.578   45.356
C   7.352   2.565   44.423
C   7.283   3.745   45.353
O   8.218   4.569   45.513
O   6.001   3.854   45.872
C   5.729   5.056   46.587
C   6.660   1.463   36.987
C   5.302   1.318   36.431
C   5.078   0.839   35.187
C   6.152   0.257   34.231
C   3.638   0.679   34.791
C   3.102   2.034   34.274
C   2.501   1.746   32.904
C   0.971   2.203   32.903
C   0.118   0.988   33.300
C   0.545   2.734   31.509
C   -0.641   3.659   31.518
C   -1.911   3.091   30.741
C   -2.124   4.095   29.606
C   -2.549   5.450   30.064
C   -3.157   3.469   28.691
C   -3.278   4.168   27.279
C   -2.311   3.556   26.234
C   -3.025   2.423   25.430
C   -2.372   1.030   25.606
C   -3.078   2.687   23.904
H   1.075   3.032   41.749
H   -0.296   -2.836   44.319
H   5.700   -3.415   46.163
H   5.376   4.700   43.221
H   3.291   4.312   41.270
H   1.940   4.680   43.428
H   3.040   6.042   42.843
H   3.315   5.085   44.272
H   7.120   3.815   41.880
H   5.982   4.608   40.755
H   5.339   1.846   40.804
H   7.152   1.964   40.809
H   -1.465   2.038   40.797
H   -1.165   3.109   42.181
H   -2.530   2.147   42.169
H   -3.476   -2.809   42.948
H   -2.332   -2.656   44.283
H   -1.797   -2.902   42.526
H   1.687   -4.884   44.094
H   3.350   -4.606   46.664
H   -0.107   -5.370   45.518
H   0.379   -4.207   46.794
H   1.203   -5.755   46.698
H   4.889   -5.253   44.191
H   3.628   -6.445   44.543
H   6.307   -6.009   45.744
H   5.136   -7.248   46.215
H   5.312   -5.697   47.045
H   8.316   -2.821   46.017
H   7.758   -2.083   47.526
H   9.136   -1.391   46.605
H   7.970   2.884   43.584
H   4.675   5.078   46.863
H   6.083   5.930   46.040
H   6.206   5.066   47.567
H   7.208   0.658   36.497
H   7.169   2.360   36.633
H   4.481   1.537   37.114
H   7.159   0.587   34.487
H   5.947   0.594   33.215
H   6.082   -0.817   34.399
H   3.080   0.322   35.657
H   3.654   -0.102   34.032
H   3.888   2.789   34.254
H   2.292   2.326   34.942
H   2.469   0.695   32.613
H   3.072   2.336   32.187
H   0.708   2.993   33.605
H   -0.853   1.132   32.825
H   0.121   1.044   34.388
H   0.469   0.036   32.903
H   0.332   1.915   30.822
H   1.341   3.336   31.071
H   -0.396   4.636   31.103
H   -0.935   3.809   32.556
H   -2.780   2.956   31.384
H   -1.740   2.143   30.231
H   -1.152   4.354   29.187
H   -1.700   6.127   29.966
H   -2.999   5.445   31.057
H   -3.348   5.871   29.454
H   -4.121   3.583   29.187
H   -3.000   2.400   28.547
H   -2.983   5.217   27.325
H   -4.301   4.146   26.903
H   -1.447   3.143   26.754
H   -1.891   4.318   25.577
H   -4.052   2.270   25.761
H   -3.196   0.372   25.331
H   -2.130   0.909   26.662
H   -1.531   0.829   24.942
H   -2.599   1.890   23.334
H   -2.646   3.649   23.627
H   -4.121   2.735   23.593

