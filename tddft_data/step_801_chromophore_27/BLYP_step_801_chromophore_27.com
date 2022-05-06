%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_801_chromophore_27 TDDFT with blyp functional

0 1
Mg   -5.856   25.061   27.041
C   -4.046   26.718   29.546
C   -6.657   22.616   29.407
C   -7.369   23.494   24.636
C   -4.363   27.305   24.622
N   -5.266   24.672   29.223
C   -4.607   25.551   30.074
C   -4.651   25.055   31.524
C   -5.334   23.625   31.412
C   -5.747   23.581   29.896
C   -4.341   22.399   31.824
C   -5.419   26.070   32.480
C   -5.031   26.029   33.941
C   -6.251   26.288   34.854
O   -6.813   27.402   35.013
O   -6.822   25.097   35.400
N   -6.835   23.294   27.048
C   -7.131   22.460   28.055
C   -8.159   21.520   27.605
C   -8.317   21.719   26.202
C   -7.475   22.878   25.914
C   -8.687   20.526   28.557
C   -9.006   20.786   25.176
O   -8.925   20.931   23.947
C   -9.659   19.514   25.607
N   -5.713   25.278   24.797
C   -6.619   24.599   24.135
C   -6.570   24.929   22.561
C   -5.729   26.280   22.573
C   -5.227   26.307   24.041
C   -6.039   23.769   21.736
C   -6.624   27.410   22.167
C   -5.894   28.492   21.268
N   -4.577   26.674   27.005
C   -4.108   27.487   25.967
C   -3.253   28.538   26.487
C   -3.297   28.379   27.923
C   -4.022   27.125   28.172
C   -2.451   29.576   25.658
C   -2.769   28.829   29.205
O   -2.054   29.739   29.533
C   -3.385   27.853   30.317
C   -2.319   27.389   31.214
O   -1.471   26.531   30.893
O   -2.396   28.083   32.418
C   -1.280   27.767   33.312
C   -8.176   25.217   35.928
C   -8.798   23.843   36.126
C   -8.719   23.034   37.237
C   -7.868   23.276   38.492
C   -9.462   21.745   37.110
C   -11.002   21.992   37.129
C   -11.763   21.016   38.104
C   -12.756   20.043   37.420
C   -14.067   20.732   37.395
C   -12.816   18.723   38.088
C   -11.970   17.629   37.326
C   -11.178   16.722   38.230
C   -9.652   17.099   38.498
C   -9.583   18.188   39.565
C   -8.791   15.827   38.860
C   -7.472   15.729   38.088
C   -6.213   15.864   38.994
C   -5.173   14.693   38.930
C   -4.979   13.937   40.252
C   -3.844   15.106   38.268
H   -7.055   21.982   30.203
H   -7.994   23.008   23.884
H   -3.939   28.015   23.909
H   -3.639   24.983   31.921
H   -6.198   23.552   32.074
H   -4.371   22.193   32.894
H   -3.306   22.605   31.554
H   -4.621   21.432   31.405
H   -6.497   26.023   32.327
H   -5.240   27.079   32.108
H   -4.367   26.872   34.126
H   -4.470   25.133   34.207
H   -9.652   20.142   28.226
H   -8.574   20.846   29.593
H   -7.957   19.718   28.521
H   -8.944   19.055   26.290
H   -9.765   18.927   24.695
H   -10.680   19.691   25.947
H   -7.636   25.069   22.385
H   -4.855   26.038   21.969
H   -5.039   23.516   22.089
H   -5.893   23.977   20.676
H   -6.664   22.926   22.032
H   -6.838   27.961   23.083
H   -7.514   27.087   21.627
H   -5.711   29.432   21.787
H   -6.516   28.635   20.384
H   -4.910   28.142   20.955
H   -1.990   30.292   26.338
H   -3.123   30.130   25.002
H   -1.700   29.067   25.053
H   -4.025   28.421   30.992
H   -1.562   28.320   34.208
H   -0.319   28.141   32.958
H   -1.316   26.690   33.473
H   -8.822   25.815   35.285
H   -8.040   25.719   36.886
H   -9.383   23.470   35.285
H   -6.970   22.664   38.412
H   -8.450   23.012   39.374
H   -7.625   24.337   38.433
H   -9.017   21.083   37.853
H   -9.187   21.421   36.106
H   -11.374   21.908   36.108
H   -11.286   23.007   37.408
H   -12.208   21.724   38.803
H   -10.911   20.521   38.571
H   -12.475   19.904   36.376
H   -14.313   20.791   38.455
H   -14.763   20.170   36.772
H   -14.005   21.783   37.114
H   -13.868   18.442   38.142
H   -12.405   18.681   39.097
H   -11.375   18.078   36.530
H   -12.752   17.056   36.828
H   -11.071   15.774   37.704
H   -11.703   16.656   39.183
H   -9.252   17.420   37.536
H   -9.072   19.062   39.163
H   -9.060   17.740   40.410
H   -10.595   18.353   39.935
H   -9.249   14.866   38.625
H   -8.537   15.706   39.913
H   -7.359   16.511   37.337
H   -7.442   14.799   37.521
H   -6.432   16.046   40.046
H   -5.745   16.817   38.750
H   -5.555   13.902   38.284
H   -3.973   13.518   40.275
H   -5.551   13.013   40.340
H   -5.194   14.459   41.184
H   -3.572   14.499   37.405
H   -3.133   15.259   39.080
H   -3.980   16.090   37.820

