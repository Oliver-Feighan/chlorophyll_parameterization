%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_551_chromophore_23 TDDFT with blyp functional

0 1
Mg   -9.829   40.515   43.632
C   -8.572   37.417   42.691
C   -7.604   42.001   41.429
C   -11.543   43.552   44.089
C   -12.551   38.856   45.231
N   -8.302   39.788   42.194
C   -7.961   38.492   41.983
C   -6.747   38.288   41.029
C   -6.449   39.821   40.633
C   -7.482   40.606   41.432
C   -4.949   40.266   40.878
C   -7.127   37.272   39.911
C   -5.917   36.257   39.587
C   -4.910   36.705   38.594
O   -4.001   37.473   38.835
O   -5.225   36.109   37.382
N   -9.443   42.582   43.077
C   -8.460   42.902   42.158
C   -8.464   44.399   42.116
C   -9.681   44.800   42.651
C   -10.279   43.599   43.368
C   -7.311   45.168   41.424
C   -10.132   46.187   42.529
O   -9.569   46.949   41.776
C   -11.233   46.771   43.395
N   -11.713   41.077   44.450
C   -12.180   42.403   44.580
C   -13.517   42.479   45.425
C   -13.863   40.977   45.481
C   -12.599   40.237   45.086
C   -13.349   43.163   46.762
C   -15.039   40.636   44.555
C   -16.211   39.724   45.073
N   -10.411   38.572   44.003
C   -11.536   38.054   44.599
C   -11.414   36.614   44.628
C   -10.219   36.312   43.897
C   -9.723   37.535   43.536
C   -12.335   35.661   45.313
C   -9.429   35.347   43.352
O   -9.526   34.132   43.406
C   -8.285   35.973   42.669
C   -7.077   35.725   43.566
O   -6.948   35.909   44.767
O   -6.013   35.341   42.786
C   -4.649   35.517   43.317
C   -4.242   36.279   36.316
C   -4.797   35.991   34.882
C   -5.717   36.701   34.182
C   -6.360   37.925   34.760
C   -6.181   36.279   32.792
C   -5.518   36.938   31.611
C   -5.390   35.945   30.398
C   -6.096   36.355   29.022
C   -5.431   35.736   27.710
C   -7.646   36.048   29.176
C   -8.541   37.295   29.048
C   -9.813   37.133   29.994
C   -9.719   37.952   31.282
C   -10.752   37.408   32.325
C   -9.966   39.470   31.089
C   -8.927   40.399   31.798
C   -8.021   41.286   30.794
C   -6.575   41.438   31.392
C   -6.540   42.665   32.260
C   -5.508   41.425   30.328
H   -6.940   42.481   40.707
H   -12.072   44.471   44.351
H   -13.503   38.469   45.601
H   -5.960   37.866   41.654
H   -6.797   39.945   39.608
H   -4.919   40.745   41.857
H   -4.652   40.915   40.055
H   -4.192   39.483   40.928
H   -7.346   37.783   38.974
H   -8.143   36.891   40.013
H   -6.302   35.270   39.332
H   -5.328   36.196   40.502
H   -7.855   45.673   40.625
H   -6.581   44.575   40.872
H   -6.898   45.885   42.133
H   -10.883   46.838   44.425
H   -12.121   46.154   43.252
H   -11.445   47.760   42.987
H   -14.342   42.920   44.865
H   -14.212   40.769   46.493
H   -12.320   43.187   47.121
H   -14.020   42.774   47.528
H   -13.799   44.155   46.822
H   -14.596   40.199   43.660
H   -15.520   41.522   44.142
H   -16.993   40.483   45.081
H   -16.237   39.298   46.076
H   -16.400   39.017   44.265
H   -11.816   34.705   45.377
H   -13.291   35.531   44.806
H   -12.648   36.007   46.298
H   -8.249   35.504   41.686
H   -4.055   35.865   42.472
H   -4.176   34.572   43.584
H   -4.633   36.105   44.235
H   -3.351   35.656   36.399
H   -3.858   37.298   36.356
H   -4.379   35.110   34.395
H   -6.052   38.774   34.150
H   -6.036   38.049   35.793
H   -7.445   37.833   34.805
H   -7.247   36.501   32.746
H   -6.184   35.189   32.794
H   -4.537   37.276   31.945
H   -6.166   37.766   31.323
H   -5.746   34.964   30.713
H   -4.317   35.891   30.215
H   -6.039   37.433   28.871
H   -5.983   34.869   27.347
H   -4.390   35.477   27.904
H   -5.396   36.538   26.973
H   -7.878   35.448   30.056
H   -7.908   35.477   28.285
H   -8.844   37.235   28.003
H   -8.043   38.260   29.141
H   -9.849   36.140   30.441
H   -10.833   37.271   29.634
H   -8.737   37.935   31.754
H   -10.818   36.322   32.262
H   -11.706   37.886   32.102
H   -10.457   37.633   33.351
H   -10.982   39.689   31.417
H   -9.956   39.631   30.011
H   -8.305   39.755   32.419
H   -9.512   41.048   32.449
H   -8.543   42.225   30.611
H   -7.880   40.711   29.879
H   -6.438   40.577   32.047
H   -7.539   43.048   32.475
H   -6.048   43.507   31.775
H   -5.904   42.484   33.127
H   -4.763   42.211   30.454
H   -5.907   41.506   29.317
H   -4.932   40.500   30.317

