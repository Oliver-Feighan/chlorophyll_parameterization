%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1001_chromophore_23 TDDFT with blyp functional

0 1
Mg   -10.768   39.740   42.765
C   -9.008   36.920   41.634
C   -8.556   41.590   40.953
C   -12.901   42.357   43.162
C   -13.376   37.608   43.910
N   -9.010   39.199   41.290
C   -8.408   38.019   41.085
C   -7.081   38.086   40.189
C   -6.834   39.672   40.285
C   -8.198   40.224   40.834
C   -5.630   40.078   41.129
C   -7.322   37.552   38.744
C   -6.542   36.338   38.238
C   -5.944   36.521   36.900
O   -5.040   37.337   36.644
O   -6.345   35.519   36.021
N   -10.733   41.704   42.190
C   -9.795   42.236   41.407
C   -10.051   43.706   41.239
C   -11.151   43.985   42.059
C   -11.664   42.654   42.524
C   -9.269   44.575   40.396
C   -11.609   45.401   42.323
O   -11.063   46.389   41.866
C   -12.910   45.685   43.115
N   -12.832   39.935   43.561
C   -13.510   41.152   43.609
C   -14.907   40.951   44.204
C   -15.076   39.401   44.290
C   -13.662   38.952   43.862
C   -15.103   41.777   45.508
C   -16.260   38.838   43.373
C   -17.051   37.629   43.927
N   -11.051   37.714   42.936
C   -12.152   36.987   43.464
C   -11.838   35.559   43.420
C   -10.595   35.496   42.657
C   -10.118   36.827   42.433
C   -12.722   34.421   43.915
C   -9.635   34.633   42.123
O   -9.608   33.384   42.106
C   -8.580   35.536   41.444
C   -7.241   35.219   41.872
O   -6.889   35.293   43.045
O   -6.667   34.510   40.869
C   -5.373   33.858   41.080
C   -5.560   35.550   34.721
C   -6.519   35.373   33.531
C   -7.305   36.245   32.852
C   -7.477   37.729   33.278
C   -7.865   35.791   31.507
C   -6.939   36.022   30.286
C   -7.465   36.847   29.032
C   -6.861   38.236   28.976
C   -5.556   38.382   28.098
C   -7.958   39.222   28.573
C   -8.755   39.840   29.838
C   -10.183   40.179   29.397
C   -11.251   38.985   29.577
C   -11.984   38.686   28.205
C   -12.150   39.221   30.760
C   -12.132   38.128   31.807
C   -13.332   38.323   32.760
C   -14.537   37.310   32.612
C   -15.747   37.976   31.917
C   -14.900   36.591   33.946
H   -7.790   42.276   40.586
H   -13.515   43.260   43.157
H   -14.199   36.911   44.080
H   -6.290   37.478   40.628
H   -6.658   40.124   39.309
H   -4.885   40.595   40.525
H   -5.140   39.293   41.705
H   -5.959   40.920   41.738
H   -7.424   38.392   38.057
H   -8.346   37.183   38.696
H   -7.213   35.495   38.074
H   -5.680   36.236   38.897
H   -8.742   43.906   39.716
H   -8.548   45.025   41.078
H   -9.891   45.328   39.910
H   -12.874   45.194   44.087
H   -13.785   45.385   42.538
H   -13.060   46.751   43.286
H   -15.700   41.463   43.660
H   -15.271   39.064   45.309
H   -14.133   42.107   45.877
H   -15.635   41.140   46.216
H   -15.681   42.655   45.221
H   -15.822   38.573   42.410
H   -16.871   39.657   42.992
H   -18.013   37.864   44.383
H   -16.445   36.999   44.578
H   -17.177   37.017   43.034
H   -12.233   33.449   43.863
H   -13.651   34.392   43.345
H   -13.061   34.499   44.949
H   -8.675   35.276   40.390
H   -4.673   34.410   41.705
H   -4.878   33.861   40.109
H   -5.552   32.937   41.635
H   -4.770   34.799   34.733
H   -5.067   36.505   34.537
H   -6.224   34.505   32.941
H   -6.941   37.817   34.223
H   -8.514   38.045   33.398
H   -7.045   38.363   32.504
H   -8.806   36.319   31.355
H   -8.112   34.749   31.711
H   -6.720   34.993   30.000
H   -6.017   36.454   30.676
H   -8.519   37.087   29.173
H   -7.422   36.215   28.145
H   -6.639   38.680   29.946
H   -5.688   38.930   27.165
H   -5.212   37.361   27.936
H   -4.879   39.021   28.664
H   -8.589   38.728   27.834
H   -7.486   40.112   28.155
H   -8.302   40.746   30.242
H   -8.830   39.248   30.750
H   -10.189   40.624   28.402
H   -10.465   41.038   30.005
H   -10.669   38.096   29.819
H   -11.876   37.639   27.923
H   -11.564   39.364   27.463
H   -13.020   38.974   28.385
H   -13.204   39.262   30.487
H   -11.834   40.112   31.303
H   -11.279   38.119   32.486
H   -12.149   37.153   31.322
H   -13.777   39.317   32.799
H   -12.958   38.306   33.783
H   -14.197   36.524   31.938
H   -16.516   37.968   32.690
H   -16.133   37.324   31.133
H   -15.660   39.023   31.627
H   -15.971   36.408   34.025
H   -14.695   37.162   34.851
H   -14.508   35.579   34.047

