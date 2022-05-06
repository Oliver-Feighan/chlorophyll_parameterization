%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_751_chromophore_27 TDDFT with cam-b3lyp functional

0 1
Mg   -5.573   24.878   26.729
C   -3.791   26.673   29.242
C   -6.560   22.653   29.117
C   -6.836   23.077   24.284
C   -3.825   26.867   24.349
N   -5.091   24.728   28.860
C   -4.506   25.634   29.754
C   -4.879   25.347   31.233
C   -5.619   23.929   31.136
C   -5.756   23.681   29.581
C   -4.814   22.885   31.879
C   -5.717   26.372   32.002
C   -5.335   26.522   33.503
C   -6.378   26.496   34.586
O   -6.371   27.338   35.463
O   -7.211   25.302   34.564
N   -6.679   23.132   26.734
C   -6.968   22.285   27.803
C   -7.692   21.193   27.345
C   -7.826   21.310   25.949
C   -7.111   22.568   25.549
C   -8.112   20.098   28.313
C   -8.412   20.385   24.946
O   -8.582   20.613   23.722
C   -9.162   19.194   25.337
N   -5.124   24.757   24.575
C   -5.938   24.063   23.861
C   -5.525   24.249   22.338
C   -5.014   25.703   22.312
C   -4.645   25.816   23.808
C   -4.630   23.101   21.734
C   -6.152   26.694   21.904
C   -6.140   27.096   20.406
N   -4.173   26.567   26.739
C   -3.566   27.235   25.641
C   -2.778   28.302   26.219
C   -2.803   28.076   27.644
C   -3.689   27.056   27.886
C   -2.083   29.474   25.529
C   -2.319   28.623   28.894
O   -1.503   29.507   29.158
C   -2.927   27.748   29.981
C   -1.850   27.042   30.712
O   -0.808   26.630   30.200
O   -2.130   27.018   32.034
C   -1.052   26.593   32.860
C   -7.838   25.245   35.864
C   -8.725   24.016   35.834
C   -8.629   22.888   36.594
C   -7.456   22.558   37.565
C   -9.770   21.898   36.601
C   -11.161   22.202   37.187
C   -11.771   21.210   38.187
C   -12.858   20.316   37.688
C   -14.263   20.976   37.619
C   -12.925   18.900   38.361
C   -11.908   17.912   37.704
C   -11.234   16.907   38.748
C   -9.692   16.876   38.643
C   -8.980   17.809   39.634
C   -9.173   15.435   38.979
C   -7.679   15.395   38.571
C   -6.875   14.879   39.850
C   -5.700   13.934   39.467
C   -5.321   13.135   40.704
C   -4.488   14.724   38.939
H   -6.894   21.937   29.871
H   -7.143   22.524   23.394
H   -3.288   27.605   23.749
H   -3.952   25.181   31.781
H   -6.525   23.863   31.739
H   -4.343   23.406   32.713
H   -4.071   22.396   31.250
H   -5.454   22.074   32.226
H   -6.755   26.071   31.861
H   -5.606   27.346   31.525
H   -4.692   27.391   33.641
H   -4.592   25.753   33.716
H   -9.118   19.714   28.143
H   -8.176   20.519   29.317
H   -7.487   19.205   28.308
H   -8.763   18.800   26.272
H   -9.194   18.392   24.601
H   -10.146   19.611   25.555
H   -6.419   24.181   21.718
H   -4.059   25.785   21.793
H   -4.344   22.393   22.512
H   -3.757   23.454   21.186
H   -5.172   22.492   21.010
H   -6.031   27.548   22.570
H   -7.107   26.299   22.251
H   -5.253   26.745   19.878
H   -6.101   28.185   20.380
H   -7.063   26.740   19.950
H   -1.053   29.155   25.373
H   -2.138   30.413   26.081
H   -2.520   29.625   24.542
H   -3.498   28.355   30.683
H   -1.321   26.739   33.906
H   -0.047   26.952   32.640
H   -0.945   25.510   32.805
H   -8.339   26.138   36.237
H   -7.130   25.026   36.663
H   -9.610   24.120   35.206
H   -7.093   23.451   38.074
H   -6.623   22.039   37.090
H   -7.888   22.042   38.423
H   -9.512   20.880   36.892
H   -10.046   21.902   35.547
H   -11.862   22.295   36.358
H   -11.063   23.165   37.687
H   -12.157   21.798   39.020
H   -10.965   20.618   38.621
H   -12.687   20.149   36.624
H   -14.306   21.986   38.028
H   -15.089   20.555   38.192
H   -14.650   21.068   36.604
H   -13.924   18.473   38.450
H   -12.533   19.096   39.359
H   -11.148   18.400   37.094
H   -12.442   17.165   37.116
H   -11.518   15.857   38.690
H   -11.622   17.164   39.733
H   -9.452   17.048   37.594
H   -8.315   17.393   40.391
H   -9.690   18.346   40.263
H   -8.421   18.633   39.190
H   -9.755   14.774   38.336
H   -9.378   15.282   40.038
H   -7.273   16.344   38.223
H   -7.581   14.619   37.811
H   -7.554   14.325   40.499
H   -6.542   15.845   40.230
H   -6.007   13.262   38.666
H   -4.311   12.758   40.542
H   -6.076   12.374   40.902
H   -5.254   13.718   41.623
H   -3.650   14.744   39.635
H   -4.854   15.749   38.873
H   -4.271   14.306   37.956

