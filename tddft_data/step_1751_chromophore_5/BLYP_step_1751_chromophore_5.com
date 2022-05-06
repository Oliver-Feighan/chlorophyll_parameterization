%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1751_chromophore_5 TDDFT with blyp functional

0 1
Mg   25.073   -7.785   47.464
C   27.418   -5.399   46.514
C   22.977   -6.789   45.043
C   23.413   -10.822   47.638
C   27.618   -9.194   49.635
N   25.293   -6.316   45.935
C   26.310   -5.425   45.683
C   26.009   -4.604   44.470
C   24.487   -4.840   44.193
C   24.208   -6.093   45.123
C   23.528   -3.588   44.689
C   26.982   -4.908   43.243
C   27.802   -3.734   42.618
C   27.560   -3.197   41.114
O   28.365   -2.730   40.286
O   26.153   -3.211   40.853
N   23.281   -8.609   46.625
C   22.554   -7.998   45.715
C   21.341   -8.849   45.437
C   21.378   -10.028   46.210
C   22.693   -9.861   46.901
C   20.233   -8.479   44.310
C   20.259   -11.025   46.342
O   19.222   -11.043   45.723
C   20.374   -11.994   47.479
N   25.551   -9.714   48.426
C   24.553   -10.641   48.449
C   24.695   -11.617   49.692
C   26.008   -11.078   50.379
C   26.500   -10.015   49.435
C   23.450   -11.693   50.570
C   27.048   -12.251   50.576
C   27.694   -12.340   51.964
N   27.073   -7.312   48.014
C   27.903   -7.989   48.895
C   29.167   -7.266   48.973
C   29.036   -6.219   48.099
C   27.741   -6.294   47.512
C   30.374   -7.628   49.780
C   29.711   -5.073   47.463
O   30.814   -4.624   47.772
C   28.667   -4.522   46.401
C   28.442   -2.959   46.646
O   27.705   -2.571   47.513
O   29.129   -2.075   45.831
C   28.566   -0.705   45.664
C   25.692   -2.874   39.495
C   25.475   -4.251   38.797
C   25.834   -4.781   37.594
C   26.783   -4.212   36.557
C   25.099   -5.946   37.031
C   23.867   -5.611   36.133
C   24.047   -6.009   34.629
C   23.488   -5.009   33.628
C   23.128   -5.782   32.280
C   24.336   -3.717   33.299
C   25.561   -3.790   32.329
C   25.506   -2.795   31.195
C   25.259   -3.503   29.824
C   24.280   -2.736   29.005
C   26.614   -3.754   29.104
C   26.683   -5.165   28.501
C   26.143   -5.169   27.061
C   24.690   -5.785   27.100
C   24.722   -7.338   27.048
C   23.841   -5.123   25.930
H   22.272   -6.478   44.269
H   22.912   -11.789   47.723
H   28.449   -9.580   50.229
H   26.095   -3.550   44.732
H   24.201   -5.077   43.169
H   24.041   -2.630   44.781
H   23.090   -3.680   45.683
H   22.736   -3.391   43.967
H   26.509   -5.410   42.399
H   27.647   -5.725   43.525
H   28.868   -3.810   42.832
H   27.471   -2.808   43.088
H   19.224   -8.431   44.721
H   20.262   -9.312   43.608
H   20.404   -7.586   43.710
H   20.800   -11.548   48.378
H   21.103   -12.734   47.152
H   19.434   -12.486   47.728
H   24.939   -12.585   49.254
H   25.766   -10.607   51.332
H   22.653   -11.181   50.031
H   23.424   -11.128   51.502
H   23.235   -12.756   50.678
H   27.858   -12.152   49.854
H   26.596   -13.235   50.454
H   27.136   -13.007   52.622
H   27.529   -11.395   52.481
H   28.775   -12.469   52.004
H   30.983   -8.195   49.075
H   30.220   -8.158   50.720
H   30.941   -6.714   49.956
H   29.070   -4.536   45.388
H   29.047   0.052   46.284
H   27.589   -0.606   46.136
H   28.573   -0.514   44.591
H   26.383   -2.277   38.900
H   24.776   -2.285   39.542
H   24.645   -4.846   39.179
H   27.688   -4.798   36.398
H   27.093   -3.206   36.841
H   26.287   -4.210   35.587
H   24.928   -6.753   37.743
H   25.757   -6.450   36.323
H   23.641   -4.548   36.218
H   22.950   -6.064   36.510
H   23.496   -6.924   34.412
H   25.089   -6.186   34.365
H   22.471   -4.741   33.913
H   23.037   -5.025   31.501
H   22.148   -6.256   32.332
H   23.912   -6.431   31.889
H   24.697   -3.422   34.285
H   23.554   -3.021   32.998
H   25.675   -4.821   31.995
H   26.456   -3.594   32.920
H   26.458   -2.265   31.233
H   24.639   -2.165   31.398
H   24.770   -4.452   30.047
H   23.272   -3.042   29.286
H   24.341   -2.902   27.929
H   24.447   -1.674   29.187
H   27.337   -3.534   29.888
H   26.705   -2.982   28.340
H   26.225   -5.932   29.125
H   27.745   -5.412   28.494
H   26.723   -5.717   26.319
H   26.142   -4.131   26.728
H   24.203   -5.469   28.023
H   24.932   -7.651   26.025
H   23.746   -7.767   27.275
H   25.436   -7.834   27.705
H   22.816   -5.467   26.065
H   24.271   -5.606   25.053
H   24.042   -4.062   25.778

