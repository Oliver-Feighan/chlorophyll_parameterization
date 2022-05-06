%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1051_chromophore_5 TDDFT with cam-b3lyp functional

0 1
Mg   24.701   -7.334   45.178
C   27.310   -5.227   43.911
C   22.536   -5.544   43.376
C   22.632   -9.942   45.645
C   27.381   -9.298   46.568
N   24.920   -5.554   43.861
C   26.031   -4.860   43.509
C   25.680   -3.817   42.446
C   24.156   -3.761   42.430
C   23.857   -5.047   43.295
C   23.561   -2.508   43.175
C   26.359   -4.129   41.130
C   26.945   -2.988   40.424
C   27.629   -3.354   39.106
O   28.838   -3.385   38.937
O   26.713   -3.613   38.130
N   22.818   -7.628   44.648
C   22.045   -6.741   43.963
C   20.729   -7.164   43.926
C   20.687   -8.434   44.682
C   22.101   -8.769   45.032
C   19.583   -6.384   43.460
C   19.403   -9.159   45.011
O   18.295   -8.779   44.758
C   19.412   -10.482   45.806
N   24.927   -9.280   46.264
C   23.878   -10.179   46.217
C   24.318   -11.543   46.702
C   25.818   -11.212   47.143
C   26.084   -9.855   46.584
C   23.226   -12.215   47.654
C   26.809   -12.358   46.687
C   27.916   -12.729   47.666
N   26.901   -7.343   45.243
C   27.787   -8.055   46.026
C   29.106   -7.424   46.017
C   28.961   -6.390   45.147
C   27.580   -6.316   44.772
C   30.384   -7.954   46.737
C   29.628   -5.298   44.574
O   30.836   -4.997   44.631
C   28.589   -4.535   43.712
C   28.488   -3.054   43.944
O   28.563   -2.145   43.151
O   28.286   -2.924   45.265
C   28.108   -1.528   45.692
C   27.227   -4.383   36.995
C   26.462   -5.673   36.795
C   26.810   -6.583   35.837
C   27.767   -6.420   34.684
C   25.938   -7.904   35.729
C   24.443   -7.514   35.462
C   24.038   -8.164   34.180
C   22.877   -7.266   33.500
C   21.737   -8.094   32.875
C   23.547   -6.501   32.338
C   23.969   -5.121   32.861
C   25.417   -4.759   32.286
C   25.384   -3.968   30.986
C   25.660   -2.497   31.286
C   26.209   -4.414   29.754
C   25.905   -3.647   28.366
C   25.314   -4.516   27.224
C   24.522   -3.589   26.241
C   23.170   -3.076   26.903
C   24.346   -4.196   24.825
H   21.796   -4.885   42.918
H   21.976   -10.815   45.641
H   28.189   -9.919   46.960
H   25.997   -2.867   42.877
H   23.731   -3.736   41.427
H   22.729   -2.064   42.630
H   24.264   -1.680   43.080
H   23.358   -2.765   44.215
H   25.653   -4.609   40.451
H   27.135   -4.827   41.446
H   27.746   -2.593   41.050
H   26.204   -2.238   40.148
H   19.033   -6.956   42.712
H   19.879   -5.425   43.035
H   19.053   -6.252   44.403
H   20.134   -10.251   46.589
H   19.765   -11.320   45.206
H   18.403   -10.678   46.168
H   24.335   -12.167   45.809
H   25.948   -11.116   48.221
H   23.658   -12.346   48.647
H   22.825   -13.177   47.335
H   22.314   -11.626   47.744
H   27.186   -11.984   45.735
H   26.292   -13.246   46.324
H   27.655   -12.484   48.696
H   28.845   -12.178   47.518
H   28.116   -13.797   47.584
H   30.107   -8.718   47.463
H   30.934   -7.175   47.266
H   30.972   -8.414   45.942
H   28.993   -4.764   42.726
H   27.148   -1.098   45.407
H   28.886   -0.919   45.231
H   28.124   -1.491   46.782
H   28.290   -4.541   37.180
H   27.135   -3.834   36.058
H   25.660   -5.905   37.496
H   28.163   -7.427   34.552
H   28.462   -5.612   34.913
H   27.102   -6.091   33.885
H   25.924   -8.268   36.757
H   26.345   -8.652   35.049
H   24.559   -6.437   35.339
H   23.884   -7.771   36.362
H   23.626   -9.154   34.377
H   24.876   -8.240   33.487
H   22.517   -6.510   34.198
H   20.791   -7.562   32.977
H   21.557   -9.022   33.419
H   21.900   -8.370   31.833
H   22.824   -6.261   31.558
H   24.295   -7.095   31.812
H   24.002   -5.099   33.951
H   23.338   -4.324   32.467
H   25.995   -5.636   31.995
H   25.865   -4.247   33.138
H   24.355   -3.938   30.627
H   24.909   -1.877   30.796
H   26.655   -2.198   30.957
H   25.660   -2.259   32.350
H   25.879   -5.435   29.564
H   27.248   -4.342   30.075
H   26.909   -3.283   28.150
H   25.342   -2.725   28.510
H   24.601   -5.223   27.646
H   26.107   -5.105   26.762
H   25.141   -2.702   26.104
H   22.295   -3.159   26.259
H   23.258   -2.014   27.135
H   22.873   -3.557   27.835
H   24.704   -5.222   24.740
H   24.831   -3.661   24.009
H   23.305   -4.265   24.508

