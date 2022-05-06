%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1451_chromophore_11 TDDFT with cam-b3lyp functional

0 1
Mg   52.956   24.421   44.386
C   50.123   26.452   43.839
C   51.125   21.647   43.439
C   55.809   22.510   44.266
C   54.709   27.228   45.260
N   50.843   24.119   43.594
C   49.859   25.052   43.550
C   48.491   24.460   43.173
C   48.877   22.936   43.375
C   50.384   22.884   43.413
C   48.307   22.246   44.696
C   47.907   24.814   41.820
C   48.827   24.396   40.608
C   48.163   23.639   39.415
O   47.257   24.059   38.672
O   48.679   22.366   39.352
N   53.428   22.359   43.921
C   52.488   21.400   43.589
C   53.184   20.118   43.477
C   54.538   20.246   43.781
C   54.632   21.731   43.960
C   52.365   18.871   43.090
C   55.549   19.175   43.804
O   55.238   18.000   43.599
C   57.019   19.437   44.109
N   54.922   24.877   44.511
C   55.923   23.886   44.590
C   57.319   24.556   44.521
C   56.996   26.005   45.006
C   55.440   26.074   44.874
C   58.500   23.843   45.352
C   57.796   27.209   44.344
C   59.318   27.030   44.229
N   52.546   26.421   44.632
C   53.305   27.403   45.211
C   52.510   28.514   45.551
C   51.242   28.231   44.957
C   51.344   26.921   44.379
C   53.027   29.723   46.198
C   49.931   28.727   44.662
O   49.407   29.734   45.181
C   49.126   27.588   43.915
C   48.803   28.137   42.567
O   49.546   28.300   41.600
O   47.501   28.656   42.599
C   46.945   29.175   41.321
C   48.080   21.470   38.418
C   48.974   21.531   37.191
C   48.899   20.834   35.997
C   47.642   19.968   35.675
C   50.018   20.917   35.004
C   51.157   19.889   35.238
C   51.759   19.340   34.002
C   53.244   18.842   34.195
C   54.249   19.572   33.211
C   53.387   17.322   33.975
C   52.855   16.904   32.557
C   53.867   16.151   31.618
C   53.181   14.984   30.905
C   53.035   13.736   31.721
C   53.975   14.702   29.597
C   53.079   14.198   28.483
C   53.373   12.756   28.178
C   54.000   12.470   26.808
C   55.078   13.470   26.333
C   52.811   12.444   25.804
H   50.394   20.837   43.394
H   56.674   21.853   44.158
H   55.410   28.027   45.513
H   47.821   24.741   43.986
H   48.569   22.181   42.652
H   49.133   21.999   45.363
H   47.887   21.261   44.492
H   47.551   22.775   45.277
H   47.779   25.892   41.729
H   46.935   24.328   41.731
H   49.671   23.760   40.875
H   49.284   25.304   40.215
H   51.365   19.072   42.707
H   52.242   18.303   44.012
H   52.914   18.156   42.476
H   57.532   18.539   44.456
H   56.876   20.027   45.015
H   57.367   20.027   43.262
H   57.764   24.537   43.526
H   57.247   25.988   46.067
H   59.271   23.380   44.736
H   57.995   23.037   45.884
H   58.977   24.576   46.003
H   57.468   28.066   44.933
H   57.328   27.427   43.384
H   59.849   27.466   45.075
H   59.781   27.403   43.315
H   59.557   25.968   44.292
H   52.972   30.423   45.365
H   54.067   29.586   46.494
H   52.282   30.105   46.897
H   48.212   27.446   44.491
H   45.895   29.281   41.592
H   47.140   28.395   40.585
H   47.350   30.161   41.091
H   47.036   21.684   38.189
H   48.209   20.440   38.751
H   49.839   22.180   37.328
H   47.312   20.110   34.646
H   46.895   20.192   36.437
H   48.094   19.001   35.898
H   50.483   21.898   35.107
H   49.679   20.896   33.969
H   50.740   19.061   35.812
H   52.004   20.317   35.775
H   51.699   20.099   33.223
H   51.042   18.557   33.755
H   53.602   19.168   35.171
H   54.676   20.449   33.696
H   53.818   19.993   32.302
H   54.995   18.832   32.924
H   52.778   16.743   34.669
H   54.410   17.008   34.181
H   52.496   17.799   32.049
H   52.059   16.192   32.773
H   54.577   15.703   32.314
H   54.470   16.760   30.943
H   52.160   15.247   30.631
H   52.920   13.878   32.796
H   53.830   13.035   31.469
H   52.076   13.296   31.446
H   54.744   13.974   29.858
H   54.481   15.615   29.286
H   53.314   14.822   27.620
H   52.015   14.299   28.698
H   52.531   12.087   28.357
H   54.052   12.319   28.911
H   54.295   11.439   27.004
H   55.504   14.086   27.125
H   54.658   14.209   25.651
H   55.847   12.975   25.740
H   53.122   12.768   24.811
H   51.954   13.061   26.076
H   52.477   11.414   25.676

