% Example of how to use FluxQubit.m:

% Define a FLuxQubit object by calling the constructor with, for example

jj3=FluxQubit("flux3JJL");

%valid circuit names are, so far, "fluxRF", "flux3JJ", "flux3JJL", "flux4JJL"
%"fluxRF2loops", "LCres" and "squidres" (to be enetered with the quotes)

%The FluxQubit properities are:

% - circuitname: name of the circuit, like "fluxRF"
% - type: "flux" or "charge"
% - parameters: physical parameters
% - fluxb: names and values of the flux biases
% - fluxb: names and values of the voltage biases
% - sweeps: contains the definitions of parameter/flux bias sweeps
% - inductors: contains the positions and the names of the inductances in
% the circuit
% - closures: contains the positions of the closure branches and the names of
% the relative flux biases
% - modes: contains an ordered list with the type (Harmonic oscillator,
% Island or Josephson) of each mode) 
% - truncations: contains the truncation values for each mode, i.e the values
% Nmax of the highest excited state considered (which corresponds to the
% number state Nmax-1 for the harmonic oscillator modes and to the state
% with charge 2eNmax for Island and Josephson modes. NB. in thi latter case
% the states Q=2e(-Nmax,-Nmax+1,...,Nmax-1,Nmax) are considered, for a
% total of 2Nmax+1 states.)
% - oplist: A list of the operators available for the circuit (to get
% expectation values of)
% - rotation: contains the matrix which specifies the modes rotation. The
% rotation should separate modes of different kind. If this value is set
% using the command jj3.rotation="Auto", the rotation matrix is defined
% automatically.
% - ioperators: names and coordinates of the branch current operators
% - parammat: contains the 3 matrices Cmat (capacitance matrix), Lmat
% (inverse inductance matrix) and Ejmat (Josephson energies matrix).
% - extraparams: contains the dependent parameters CJ (intrinsic
% capacitances of each junction) and EJ (Josephon energy)

%Visualise a graph of the circuit:

jj3.circuitgraph;

%View the parameters by calling the object property "parameters":

jj3.parameters

%Change a parameter with, for instance

jj3.parameters.alpha=0.55;

%Calculate the energy spectrum for the given set of parameters by calling
%the method calculateE

out=jj3.calculateE(10,'Relative',true);

%The first input is the number of output levels, the option 'Relative'
%allows to subtract the ground state energy when set to true (false is the
%defualt option, so one can just write out=jj3.calculateE(10)

%Define a parameter sweep:

jj3.paramsweep("Asweep",'alpha',0.3:0.01:0.8);

%the first argument is the sweep name, the second is the parameter to be
%sweeped, the third is the range of values. Sweeps of more than one
%variable can be defined using cells:

jj3.paramsweep("Sweep2params",{'alpha','Ic'},{0.3:0.01:0.8,3e-7:1e-8:8e-7});

%the ranges of the two parameters have to be of equal size, otherwise an
%error is returned.

%For a flux sweep, use the fluxsweep method, instead:

jj3.fluxsweep("fluxsweep",'fz',0.45:0.002:0.55);

% To calculate the spectrum as a function of the sweeped parameter(s) of
% flux(es), use the sweepE method:

out2=jj3.sweepE("Asweep",3,'Relative',true,'Plot',true);
out3=jj3.sweepE("Sweep2params",3,'Relative',true,'Plot',true);
out4=jj3.sweepE("fluxsweep",10,'Relative',true,'Plot',true);

% The first input is the sweep name defined before and the second is the
% number of output levels. The 'Plot' option enables the final plot of the
% results. NB: the original circuit properties/biases are restored at the
% end of the sweep.

% The FluxQubit code can also calculate expectation values of various
% current/charge operators (listen in the property oplist) between two
% states, for instance, the following command calculates the expectation
% value of the loop current on the ground state. Currents are expressed in
% nA and charges are given in terms of the corresponding voltage Q/C in nV.

jj3.fluxb.fz=0.5;
jj3.matelem('Iz',1,1)

%the first input parameter is the operator name and is followed by the two
%states indices, with 1 indicating the ground state.
%As expected, the current expectation value at the optimal bias point is 0.
%The size of the persistent current of the qubit states is, by definition:

abs(jj3.matelem('Iz',1,2))

%The expectation value of an operator can also be calculated as a function
%of a swept parameter with the sweepOp method:

jj3.sweepOp('Iz',1,1,"fluxsweep",'Plot',1);

%in this case the third input is the sweep name and the final option
%activates the plot.

%To clear the sweeps, do

jj3.sweeps={};

%Here's another example with a 3JJ qubit:
qubit=FluxQubit("flux3JJL");
qubit.fluxsweep("fluxsweep",'fz',0.49:0.0002:0.51);
out5=qubit.sweepOp('Iz',1,1,"fluxsweep",'Plot',1);
out6=qubit.sweepE("fluxsweep",10,'Relative',true,'Plot',true);

%Pauli coefficients:

%bias the qubit at degeneracy:
qubit.fluxb.fz=0.5;

%Use coeffP to determine the coefficient in front of a specific Pauli
%operator in the effective Hamiltonian of the qubit (at a fixed parameter
%configuration). The first input is a string or a cell list of strings
%representing aliases for each Pauli operator we are interest in, for
%instance {"x","z"} to get the coefficeints of sigma_x and sigma_z. The
%second input is an index specifying the method to do the calculation.
%Methods 1 and 2 are more approximated, so they are deprecated. Method 3 is
%based on diagonalising the current operator to find the computational 
%basis states given the qubit parameter configuration. The result is
%returned as a numerical array.

qubit.coeffP("x",3)
qubit.coeffP({"x","y","z"},3)

%Use sweepcoeffP to determine the coefficient in front of a specific Pauli
%operator in the effective Hamiltonian of the qubit, as a function of a
%sweeped paremter. The first input is the sweep name defined before,
%the second is string or a cell list of strings representing aliases
%for each Pauli operator we are interest in, for instance {"x","z"} to get
%the coefficeints of sigma_x and sigma_z. The third input is an index
%specifying the method to do the calculation. '3' should be the preferred
%option. Input 4 is the 'Plot' option, which enables the final plot of the
%results if input 5 is set to true. NB: the original circuit
%properties/biases are restored at the end of the sweep.

out7=qubit.sweepcoeffP("fluxsweep","x",3,'Plot',true);
out8=qubit.sweepcoeffP("fluxsweep",{"x","y","z"},3,'Plot',true);

%Finally, let us consider a charge qubit in the form of a Cooper pair box:

qubit=FluxQubit("SCP");

%voltage biases are specified in the voltb property, which contains node,
%the node to which the voltage is applied, Cg, the gate capacitance adn Qg,
%the gate charge (in units of a Cooper pair charge). voltb is initialised
%when the qubit class is created.

qubit.voltb
qubit.calculateE(10,'Relative',1)'

%A voltage sweep can be defined using the voltsweep method:
qubit.voltsweep("vsweep",1,linspace(0.45,0.55,101));

%To get the correct results, the offset charge on the sweeped node must be
%set to zero (once) before running sweepE, sweepOp and sweepcoeffP:

qubit.voltb.Qg=0;
qubit.sweepE("vsweep",10,'Relative',1,'Plot',1);
axis([xlim 0 6]);

%The Pauli coefficients, as a function of the applied voltage, can be
%obtained using the sweepcoeffP method:
out9=qubit.sweepcoeffP("vsweep",{'x','y','z'},3,'Plot',true);
%Notice that currently only reduction method 3 (local reduction) works