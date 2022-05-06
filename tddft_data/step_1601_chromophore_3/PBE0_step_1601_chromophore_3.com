%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1601_chromophore_3 TDDFT with PBE1PBE functional

0 1
Mg   1.657   8.184   25.970
C   2.152   10.269   28.814
C   2.095   5.448   28.233
C   1.644   6.090   23.322
C   1.526   10.920   23.976
N   2.127   7.902   28.424
C   2.210   8.972   29.280
C   2.042   8.389   30.692
C   2.246   6.838   30.477
C   2.063   6.662   28.941
C   3.700   6.316   30.921
C   0.625   8.666   31.369
C   0.622   9.019   32.858
C   1.630   8.534   33.821
O   2.640   9.135   34.061
O   1.393   7.304   34.364
N   1.814   6.109   25.806
C   1.965   5.186   26.830
C   2.138   3.886   26.259
C   1.987   4.005   24.868
C   1.800   5.466   24.601
C   2.445   2.641   27.139
C   2.019   2.787   23.891
O   1.938   3.022   22.688
C   2.075   1.329   24.226
N   1.664   8.491   23.938
C   1.540   7.462   23.024
C   1.175   7.996   21.627
C   1.185   9.528   21.788
C   1.387   9.718   23.345
C   2.052   7.464   20.452
C   -0.078   10.254   21.280
C   0.180   11.454   20.335
N   1.841   10.199   26.288
C   1.780   11.213   25.362
C   2.013   12.536   26.014
C   2.134   12.163   27.335
C   2.117   10.729   27.479
C   1.882   13.938   25.406
C   2.328   12.763   28.706
O   2.560   13.922   28.981
C   2.130   11.512   29.673
C   3.309   11.627   30.672
O   4.494   11.323   30.441
O   2.879   12.034   31.869
C   3.999   12.149   32.860
C   2.273   7.003   35.488
C   1.809   5.767   36.179
C   2.258   5.206   37.323
C   3.410   5.632   38.156
C   1.466   4.067   38.000
C   1.974   2.606   37.695
C   0.841   1.573   37.340
C   0.564   0.573   38.435
C   -0.978   0.292   38.325
C   1.333   -0.737   38.333
C   2.258   -0.818   39.580
C   3.475   -1.784   39.286
C   3.480   -2.987   40.264
C   4.695   -3.040   41.184
C   3.417   -4.276   39.439
C   2.040   -4.616   38.717
C   1.576   -6.060   38.867
C   2.035   -7.005   37.779
C   0.707   -7.454   37.072
C   2.831   -8.140   38.285
H   2.178   4.638   28.961
H   1.441   5.492   22.432
H   1.504   11.828   23.370
H   2.833   8.800   31.319
H   1.542   6.302   31.114
H   4.472   7.079   31.026
H   4.140   5.740   30.107
H   3.841   5.603   31.734
H   -0.018   7.788   31.311
H   0.140   9.410   30.737
H   -0.240   8.534   33.317
H   0.452   10.078   33.054
H   3.000   1.889   26.578
H   1.470   2.265   27.450
H   3.127   2.989   27.914
H   3.090   1.287   24.620
H   2.028   0.674   23.356
H   1.340   0.889   24.900
H   0.180   7.582   21.463
H   2.082   9.982   21.367
H   1.664   6.492   20.146
H   3.081   7.380   20.800
H   2.094   8.120   19.583
H   -0.787   10.528   22.061
H   -0.585   9.528   20.644
H   0.007   11.019   19.350
H   1.192   11.850   20.420
H   -0.562   12.227   20.532
H   1.013   14.424   25.849
H   1.677   13.742   24.353
H   2.815   14.480   25.555
H   1.117   11.532   30.074
H   4.792   12.826   32.544
H   4.397   11.147   33.024
H   3.728   12.495   33.858
H   2.214   7.725   36.303
H   3.332   6.860   35.272
H   0.929   5.298   35.740
H   3.927   6.501   37.748
H   4.074   4.768   38.136
H   3.151   5.941   39.169
H   0.397   4.092   37.788
H   1.322   4.276   39.060
H   2.580   2.298   38.547
H   2.616   2.720   36.822
H   1.183   1.121   36.409
H   -0.007   2.175   37.011
H   0.666   1.073   39.399
H   -1.474   0.782   37.487
H   -1.458   0.584   39.259
H   -1.169   -0.761   38.118
H   1.911   -0.740   37.409
H   0.670   -1.601   38.285
H   1.683   -1.074   40.470
H   2.696   0.135   39.876
H   4.351   -1.135   39.278
H   3.367   -2.174   38.274
H   2.654   -2.861   40.963
H   5.376   -3.884   41.075
H   4.266   -3.192   42.174
H   5.345   -2.171   41.082
H   3.519   -5.052   40.197
H   4.226   -4.411   38.722
H   2.099   -4.405   37.649
H   1.350   -3.887   39.141
H   0.491   -5.956   38.862
H   1.883   -6.576   39.777
H   2.595   -6.530   36.974
H   0.856   -8.207   36.298
H   0.226   -6.653   36.510
H   0.090   -8.039   37.755
H   2.282   -8.872   38.877
H   3.653   -7.715   38.861
H   3.256   -8.675   37.435

