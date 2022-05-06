%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1801_chromophore_22 TDDFT with blyp functional

0 1
Mg   8.233   47.703   26.118
C   5.933   47.610   28.792
C   10.682   48.442   28.375
C   10.372   48.459   23.568
C   5.516   47.851   23.926
N   8.258   48.116   28.393
C   7.229   47.969   29.287
C   7.776   48.116   30.719
C   9.298   48.699   30.535
C   9.454   48.475   29.025
C   9.529   50.236   31.005
C   7.563   46.933   31.743
C   8.696   46.661   32.823
C   8.187   45.952   34.108
O   7.504   44.956   34.209
O   8.696   46.613   35.167
N   10.289   48.081   25.983
C   11.083   48.299   27.031
C   12.502   48.325   26.511
C   12.413   48.225   25.054
C   10.975   48.297   24.820
C   13.714   48.615   27.353
C   13.471   48.130   23.985
O   13.335   48.203   22.714
C   14.892   48.024   24.461
N   7.983   48.257   24.076
C   9.042   48.461   23.284
C   8.594   48.532   21.884
C   7.039   48.302   21.937
C   6.798   48.043   23.390
C   9.087   49.782   21.180
C   6.509   47.257   20.935
C   6.531   47.642   19.496
N   6.075   47.772   26.301
C   5.167   47.712   25.292
C   3.889   47.514   25.912
C   4.140   47.472   27.311
C   5.499   47.636   27.471
C   2.625   47.452   25.185
C   3.486   47.257   28.564
O   2.289   46.993   28.919
C   4.655   47.479   29.667
C   4.691   46.269   30.438
O   5.091   45.158   30.127
O   4.256   46.667   31.662
C   4.221   45.541   32.689
C   8.198   45.988   36.415
C   8.912   46.719   37.553
C   8.972   46.362   38.862
C   8.217   45.123   39.417
C   9.375   47.320   40.009
C   10.517   46.844   40.936
C   11.377   47.998   41.481
C   11.502   47.857   43.022
C   11.814   49.272   43.622
C   12.607   46.723   43.416
C   12.119   45.477   44.140
C   12.579   45.501   45.640
C   13.887   44.700   45.868
C   14.836   45.582   46.791
C   13.835   43.338   46.514
C   14.768   42.339   45.766
C   14.172   41.452   44.683
C   15.106   40.648   43.698
C   15.528   41.416   42.461
C   14.468   39.278   43.262
H   11.441   48.769   29.088
H   11.045   48.799   22.778
H   4.726   47.911   23.175
H   7.113   48.911   31.058
H   10.154   48.115   30.873
H   8.649   50.614   31.525
H   9.772   50.758   30.079
H   10.451   50.239   31.586
H   7.380   46.000   31.209
H   6.639   47.056   32.308
H   9.346   47.526   32.954
H   9.308   45.864   32.400
H   14.244   49.518   27.051
H   14.403   47.771   27.315
H   13.531   48.763   28.417
H   15.521   47.802   23.599
H   14.969   47.217   25.189
H   15.187   49.008   24.824
H   9.035   47.654   21.414
H   6.603   49.258   21.646
H   8.246   50.228   20.649
H   9.856   49.488   20.466
H   9.630   50.475   21.824
H   5.541   46.883   21.271
H   7.211   46.442   21.112
H   7.390   47.217   18.977
H   6.627   48.723   19.396
H   5.589   47.381   19.014
H   2.234   48.466   25.271
H   2.018   46.665   25.631
H   2.720   47.326   24.106
H   4.415   48.384   30.223
H   4.764   45.957   33.538
H   4.675   44.605   32.364
H   3.185   45.314   32.941
H   8.540   44.954   36.459
H   7.128   46.189   36.468
H   9.315   47.702   37.308
H   8.539   44.863   40.426
H   8.666   44.333   38.814
H   7.132   45.157   39.325
H   8.505   47.326   40.665
H   9.592   48.320   39.634
H   11.211   46.188   40.410
H   10.151   46.144   41.688
H   10.903   48.940   41.205
H   12.353   48.069   41.002
H   10.598   47.551   43.549
H   11.269   49.387   44.560
H   11.647   50.103   42.936
H   12.875   49.301   43.870
H   13.406   47.281   43.905
H   13.174   46.315   42.580
H   12.609   44.614   43.691
H   11.031   45.532   44.159
H   11.770   45.124   46.266
H   12.671   46.532   45.979
H   14.433   44.619   44.928
H   14.327   45.693   47.748
H   15.110   46.469   46.219
H   15.743   45.041   47.061
H   12.797   43.008   46.512
H   14.086   43.518   47.560
H   15.239   41.768   46.566
H   15.598   42.803   45.233
H   13.509   42.097   44.107
H   13.547   40.729   45.207
H   16.050   40.446   44.203
H   16.433   41.043   41.982
H   15.935   42.363   42.815
H   14.715   41.550   41.747
H   14.990   38.398   43.637
H   14.511   39.255   42.173
H   13.408   39.267   43.516

