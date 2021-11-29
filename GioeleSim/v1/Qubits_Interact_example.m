% Example of how to use QubitsInteract.m:

%The QubitsInteract class is used to calculate the spectrum of interacting
%superconducting circuits. Inductive and capacitive interactions are
%currently supported

%Firstly, define the different circuits in the system as FluxQubits objects
% qubit1=FluxQubit("flux3JJL");
% qubit2=FluxQubit("flux3JJL");
qubit1=FluxQubit("fluxRF");
qubit2=FluxQubit("fluxRF");
coupler=FluxQubit("fluxRF2loops");

%Secondly, load the circuits inside the QubitsInteract object. NB: the
%circuits/FluxQubit objects are effectively passed as handles, so  that
%any modification of their properties implemented within "coupled" persists
%even when this has been deleted
coupled=QubitsInteract(qubit1,qubit2,coupler);

%Define first level truncations (i.e. the number of eigenstates of the
%uncoupled hamiltonians kept for each circuit, which are used to calculate
%the interaction terms.
coupled.lev1truncation=[10,10,5];

%Define the interactions using the setinteractions method.
% - Its first two inputs are the integer indices if the interacting
% circuits (in the order in which they have been loaded, that can be
% checked by looking at the property coupled.cirucuitnames),
% - the third input defines the type of interaction, 'Ind' for inductive
% and 'Cap' for capacitive,
% - Inputs 4 and 5 specify the elements that participate in the
% interactions, namely the two inductively  coupled inductors or the two
% capacitively coupled nodes.
% - Input 6 defines the strength of the interaction, in Henry's for and
% inductive interaction and in Farads for a capacitve one.

%inductive interaction between circuits 1 and 3 (i.e. between qubit 1 and
%the coupler)
coupled.setinteractions(1,3,'Ind','Lq','Lc',10e-11);
%inductive interactions between 2 and 3 and between 3 and 4
coupled.setinteractions(2,3,'Ind','Lq','Lc',10e-11);

%Set fixed biases using the changebias method:
% - input 1: circuit index
% - input 2: name of the flux bias, e.g. 'fz' (the appropriate name
% can be checked using, for instance, qubit1.fluxb)
% - input 3: new flux bias value
% The physisical properties of the circuits can similarly be chaged using
% the changeparam method.
coupled.changebias(1,'fz',0.5001); %qubits biased near half Phi0 (coupler biases are set to 0 by default)
coupled.changebias(2,'fz',0.5002);

%Define a flux sweep (coupler fx sweep in this case) using the fluxsweep
%method:
% - input 1: sweep name
% - input 2: circuit to which the sweep is applied (or circuits, passed as
% an array)
% - input 3: name of the flux bias variable (multiple variables can be
% passed as a cell)
% - input 4: sweep range (multiple ranges - of equal length - can be passed
% as a cell)
coupled.fluxsweep("csweep",3,'fx',0:0.02:2);

%Get the eigenvaues of the coupled system as a function of the sweeped
%flux. This method has the same behaviour as the sweepE method of the
%FluxQubit class. NB: the initial flux values are restored at the end of the
%sweep
out=coupled.sweepE("csweep",10,'Relative',1,'Plot',1);

%Get the matrix element of a circuit operator between two energy
%eigenstates as a function of a sweeped flux/parameter.
Iout=coupled.sweepOp("csweep",1,'Iz',1,1,'Plot',true);

%Multiple flux sweep:
coupled.fluxsweep("fluxessweep",[1 2],{'fz','fz'},{0.49:0.002:0.51,0.49:0.002:0.51});
out2=coupled.sweepE("fluxessweep",10,'Relative',1,'Plot',1);

%Example 2: 3 coupled qubits

%Define a 3rd qubit
qubit3=FluxQubit("fluxRF");

%Load the circuits inside the QubitsInteract object
coupled2=QubitsInteract(qubit1,qubit2,qubit3,coupler);

%Define first and second level truncations
coupled2.lev1truncation=[10,10,10,5];
coupled2.lev2truncation={{{1,2},64};{{3,4},25}};
% means: subsystem 1 formed of circits 1 and 2 -> 64 levels kept, subsystem
% 2 formed of circuits 3 and 4 -> 25 levels kept

%Define interactions:
%inductive interaction between circuits 1 and 4 (i.e. between qubit 1 and
%the coupler)
coupled2.setinteractions(1,4,'Ind','Lq','Lc',1e-10);
%inductive interactions between 2 and 4 and between 3 and 4
coupled2.setinteractions(2,4,'Ind','Lq','Lc',1e-10);
coupled2.setinteractions(3,4,'Ind','Lq','Lc',1e-10);

%Set fixed biases
%NB: the first two lines down here are actually unnecessary, since the flux
%bias values for qubit1 and qubit2 have already been modified to be 0.5.
coupled2.changebias(1,'fz',0.5);
coupled2.changebias(2,'fz',0.5);
coupled2.changebias(3,'fz',0.5);

%Define a flux sweep (coupler fx sweep in this case)
coupled2.fluxsweep("csweep",4,'fx',0:0.04:2);

%Get Pauli coefficients:
%Use coeffP to determine the coefficient in front of a specific Pauli
%operator in the effective Hamiltonian of the qubit (at a fixed parameter
%configuration). The first input is a 1D or 2D cell array of strings of strings
%representing aliases for each Pauli operator we are interest in. The
%second input is a numerical array containing the indices of every
%circuit to be treated as a qubit (as opposed to a coupler). The result is
%returned as a numerical array.

%For the two Pauli operators sigma_z1*sigma_z2, sigma_z2*sigma_z3 and
% sigma_z1*sigma_z3:
coupled2.coeffP({{"z","z","I","I"};{"I","z","z","I"};{"z","I","z","I"}},[1 2 3])

%Use sweepcoeffP to determine the coefficient in front of a specific Pauli
%operator in the effective Hamiltonian of the system, as a function of a
%sweeped parameter. The first input is the sweep name defined before,
%the second is a 1D or 2D cell array of strings of strings representing aliases
%for each Pauli operator we are interest in. NB: a name needs to be
%provided for every circuit (in the order of appearance in
%obj.circuitnames). "I" for identity is used for coupler circuits.
%The third input is a numerical array containing the indices of every
%circuit to be treated as a qubit (as opposed to a coupler).
%Input 4 is the 'Plot' option, which enables the final plot of the
%results if input 5 is set to true. NB: the original circuit
%properties/biases are restored at the end of the sweep.

%To extract two-local zz interaction coefficients for every pair of qubits,
%plus the 3 local term:
coefficients={{'z','z','I','I'};{'z','I','z','I'};{'I','z','z','I'};{'z','z','z','I'}};
pauli2=coupled2.sweepcoeffP("csweep",coefficients,[1 2 3],'Plot',true);

%Get the eigenvaues of the coupled system as a function of the sweeped flux 
out3=coupled2.sweepE("csweep",30,'Relative',1,'Plot',1);
%NB: When using 2nd level hierarchical (or subsystem) diagonalisation, I
%noticed that sometimes MATLAB fails to find some of the eigenvalues,
%increasing the number of requested eigenvalues can fix this problem.