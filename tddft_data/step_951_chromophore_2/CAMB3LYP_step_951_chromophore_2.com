%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_951_chromophore_2 TDDFT with cam-b3lyp functional

0 1
Mg   2.930   0.280   44.224
C   6.135   1.819   44.066
C   1.635   2.995   42.729
C   0.244   -1.514   43.681
C   4.587   -2.539   45.547
N   3.831   2.180   43.469
C   5.137   2.562   43.462
C   5.342   3.955   42.702
C   3.837   4.399   42.561
C   3.006   3.137   42.837
C   3.418   5.689   43.335
C   6.173   3.874   41.376
C   5.796   2.678   40.511
C   6.525   2.590   39.168
O   7.481   1.895   38.857
O   5.926   3.452   38.282
N   1.121   0.710   43.430
C   0.714   1.904   42.909
C   -0.698   1.859   42.440
C   -1.138   0.567   42.822
C   0.066   -0.203   43.358
C   -1.475   3.041   41.942
C   -2.622   0.009   42.736
O   -3.559   0.726   42.221
C   -2.983   -1.357   43.309
N   2.533   -1.789   44.405
C   1.287   -2.231   44.192
C   1.276   -3.738   44.370
C   2.691   -4.091   44.898
C   3.334   -2.719   44.998
C   0.113   -4.388   45.260
C   3.558   -5.144   44.153
C   3.827   -6.501   44.919
N   4.943   -0.231   44.826
C   5.370   -1.364   45.417
C   6.802   -1.180   45.706
C   7.147   0.054   45.183
C   5.979   0.602   44.658
C   7.692   -2.238   46.325
C   8.239   0.970   45.021
O   9.453   0.845   45.290
C   7.621   2.129   44.110
C   7.795   3.527   44.814
O   8.564   4.359   44.320
O   7.004   3.686   45.878
C   7.042   4.973   46.508
C   6.443   3.668   36.976
C   5.274   3.326   36.089
C   5.050   2.192   35.392
C   5.981   1.075   35.270
C   3.804   2.085   34.591
C   3.921   2.459   33.123
C   3.325   1.328   32.261
C   1.758   1.411   32.157
C   1.138   -0.042   31.893
C   1.281   2.383   31.053
C   0.671   3.747   31.659
C   -0.829   4.016   31.207
C   -0.945   4.535   29.779
C   -1.326   6.016   29.658
C   -1.920   3.667   28.859
C   -1.449   3.475   27.419
C   -1.909   4.588   26.380
C   -2.727   4.054   25.146
C   -2.403   4.732   23.813
C   -4.245   3.974   25.372
H   1.138   3.810   42.200
H   -0.559   -2.232   43.502
H   4.998   -3.400   46.077
H   5.876   4.730   43.253
H   3.745   4.817   41.558
H   2.673   6.287   42.811
H   4.349   6.154   43.660
H   2.883   5.395   44.238
H   7.241   3.935   41.587
H   5.898   4.785   40.845
H   4.707   2.698   40.470
H   5.983   1.759   41.067
H   -1.950   2.761   41.002
H   -0.792   3.863   41.727
H   -2.164   3.516   42.641
H   -2.475   -1.389   44.273
H   -2.454   -2.057   42.663
H   -4.039   -1.596   43.435
H   1.205   -4.271   43.422
H   2.693   -4.386   45.947
H   0.454   -5.294   45.761
H   -0.721   -4.840   44.722
H   -0.212   -3.655   45.998
H   4.528   -4.744   43.862
H   3.087   -5.439   43.215
H   4.747   -6.482   45.503
H   4.072   -7.305   44.225
H   2.965   -6.725   45.547
H   7.773   -2.169   47.410
H   8.662   -2.148   45.837
H   7.214   -3.183   46.066
H   8.133   2.183   43.149
H   6.267   5.577   46.035
H   7.947   5.552   46.329
H   6.756   5.021   47.559
H   7.374   3.152   36.740
H   6.633   4.739   36.900
H   4.614   4.138   35.780
H   6.899   1.375   35.774
H   6.303   0.953   34.236
H   5.561   0.125   35.601
H   3.067   2.796   34.964
H   3.237   1.171   34.768
H   4.934   2.593   32.742
H   3.338   3.375   33.029
H   3.486   0.338   32.689
H   3.747   1.480   31.268
H   1.440   1.650   33.172
H   0.816   -0.593   32.777
H   1.877   -0.601   31.318
H   0.323   -0.024   31.169
H   0.495   1.982   30.413
H   2.139   2.696   30.458
H   1.315   4.562   31.328
H   0.720   3.727   32.748
H   -1.261   4.747   31.890
H   -1.355   3.074   31.366
H   0.009   4.576   29.253
H   -2.397   6.125   29.833
H   -1.199   6.395   28.644
H   -0.909   6.677   30.417
H   -2.895   4.153   28.838
H   -2.047   2.624   29.150
H   -1.720   2.512   26.985
H   -0.361   3.445   27.379
H   -0.997   5.081   26.044
H   -2.527   5.313   26.909
H   -2.430   3.025   24.945
H   -1.592   4.160   23.362
H   -1.920   5.709   23.847
H   -3.267   4.862   23.161
H   -4.820   4.560   24.655
H   -4.517   4.412   26.332
H   -4.507   2.916   25.348

