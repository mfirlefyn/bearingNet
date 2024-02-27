* nest location: 5,5
* grid distance: 10x10 m
* grid amount: 100
* image dims: 1800x201

* CNN arch:
	 * conv layer 1: (1,2,5,stride=4), tanh activation
	 * conv layer 2: (2,4,5,stride=4), tanh activation
	 * flatten: (x,1)
	 * fully conn layer 1: (5376,1), tanh activation
* lr = 0.9e-3, scheduler: ReduceLROnPlateau(optimizer, mode='min', factor=0.99, patience=1, min_lr=1e-12, eps=1e-14, verbose=True)
* nr. training samples: 396
* epochs: 100

note: scaled the labels /1.1 as to get rid of the boundary values for tanh