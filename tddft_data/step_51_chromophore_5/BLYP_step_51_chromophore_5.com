%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_51_chromophore_5 TDDFT with blyp functional

0 1
Mg   24.643   -6.258   45.520
C   27.137   -3.850   44.439
C   22.465   -5.020   43.355
C   22.777   -8.966   46.174
C   27.349   -7.781   47.154
N   24.823   -4.632   43.986
C   25.844   -3.699   43.810
C   25.431   -2.693   42.767
C   23.850   -2.988   42.686
C   23.666   -4.295   43.406
C   22.942   -1.896   43.285
C   26.185   -2.828   41.409
C   26.771   -1.483   40.984
C   25.821   -0.561   40.176
O   24.864   0.047   40.561
O   26.260   -0.543   38.892
N   22.798   -6.929   44.844
C   22.048   -6.239   44.008
C   20.761   -6.855   43.883
C   20.813   -8.007   44.680
C   22.184   -8.065   45.230
C   19.678   -6.127   43.178
C   19.616   -8.939   45.012
O   18.553   -8.574   44.598
C   19.625   -10.155   45.926
N   25.037   -8.143   46.392
C   24.020   -9.033   46.679
C   24.426   -10.185   47.578
C   25.966   -9.885   47.658
C   26.138   -8.516   47.062
C   23.639   -10.066   48.890
C   26.809   -10.946   46.911
C   28.068   -11.452   47.759
N   26.658   -5.815   45.964
C   27.656   -6.541   46.605
C   28.906   -5.884   46.720
C   28.735   -4.837   45.840
C   27.397   -4.879   45.372
C   30.256   -6.502   47.343
C   29.413   -3.715   45.216
O   30.574   -3.297   45.415
C   28.422   -3.115   44.246
C   28.275   -1.695   44.668
O   28.799   -0.724   44.118
O   27.368   -1.509   45.669
C   27.176   -0.127   46.133
C   25.536   0.188   37.891
C   24.467   -0.702   37.440
C   24.506   -1.940   36.858
C   25.815   -2.776   36.653
C   23.224   -2.674   36.589
C   23.106   -3.176   35.088
C   22.404   -4.503   34.958
C   22.281   -4.923   33.516
C   21.001   -5.726   33.271
C   23.478   -5.705   33.054
C   23.627   -5.553   31.514
C   24.730   -4.453   31.144
C   24.411   -3.584   29.953
C   23.698   -2.321   30.398
C   25.715   -3.225   29.202
C   25.733   -3.814   27.764
C   24.950   -3.001   26.669
C   23.929   -3.953   25.935
C   24.369   -4.262   24.500
C   22.531   -3.279   25.955
H   21.855   -4.636   42.534
H   22.198   -9.839   46.481
H   28.172   -8.214   47.726
H   25.604   -1.711   43.207
H   23.585   -3.062   41.631
H   23.539   -1.077   43.685
H   22.424   -2.287   44.161
H   22.181   -1.504   42.610
H   25.475   -3.164   40.653
H   26.999   -3.533   41.575
H   27.669   -1.781   40.442
H   26.994   -0.862   41.851
H   20.002   -5.734   42.214
H   19.354   -5.252   43.743
H   18.773   -6.714   43.019
H   20.016   -9.904   46.912
H   20.321   -10.816   45.410
H   18.569   -10.414   45.997
H   24.182   -11.135   47.103
H   26.228   -9.945   48.714
H   22.596   -10.247   48.630
H   23.920   -9.120   49.353
H   23.956   -10.969   49.411
H   27.105   -10.480   45.972
H   26.244   -11.825   46.600
H   28.042   -12.537   47.869
H   28.011   -11.200   48.818
H   29.034   -11.125   47.375
H   30.853   -5.732   47.832
H   30.764   -7.077   46.570
H   30.039   -7.351   47.991
H   28.956   -3.229   43.303
H   26.795   0.511   45.336
H   28.087   0.375   46.457
H   26.570   -0.074   47.037
H   26.250   0.305   37.075
H   25.189   1.116   38.345
H   23.504   -0.242   37.661
H   26.300   -2.974   37.609
H   26.529   -2.112   36.166
H   25.667   -3.635   36.000
H   22.448   -1.909   36.564
H   23.117   -3.485   37.310
H   24.056   -3.171   34.553
H   22.447   -2.531   34.507
H   21.420   -4.424   35.419
H   23.139   -5.136   35.456
H   22.325   -4.078   32.829
H   20.292   -5.002   32.870
H   20.577   -6.126   34.192
H   20.948   -6.529   32.536
H   23.385   -6.780   33.206
H   24.371   -5.300   33.529
H   22.705   -5.349   30.970
H   24.035   -6.513   31.199
H   25.537   -5.166   30.975
H   25.107   -3.942   32.030
H   23.602   -4.057   29.396
H   23.125   -2.405   31.321
H   23.082   -1.979   29.566
H   24.478   -1.565   30.495
H   26.610   -3.644   29.662
H   26.005   -2.178   29.115
H   25.360   -4.833   27.862
H   26.789   -3.794   27.494
H   25.747   -2.661   26.008
H   24.452   -2.103   27.035
H   23.795   -4.920   26.419
H   23.629   -3.779   23.861
H   24.348   -5.338   24.324
H   25.410   -4.022   24.285
H   21.956   -3.483   25.052
H   22.637   -2.196   26.025
H   22.048   -3.666   26.853

