% The FluxQubit class allows the calculation of the eigenspectrum of a
% number of supercondung circuits, as well as other system properties like
% the expectation values of observables, and the coefficeints of the
% underlying qubit Hamiltonian. These properties can also be calculated as 
% a function of a varying physical parameter/magnetic flux. The code is 
% based on the principles of Quantum Circuit Analysis and on the efficient 
% method of representation of the Hamiltonians suggested by J.A. Kerman 
% ("Efficient simulation of complex, Josephson quantum circuits with linear
% inductances", unpublished).

classdef FluxQubit < handle
    properties (Constant, Access=private)
        Phi0=2.067834e-15; % Physical constants
        h=6.62607004e-34;
        e=1.60217662e-19;
        Sc=60; %Fabrication constants
        Jc=3;
    end
    properties
        circuitname % Name of the cirucit
        type % Circuit type (charge,flux,coupler)
        parameters % Physical parameters
        fluxb % fluxbiases
        voltb % voltage biases
        sweeps % contains the parameters sweeps defied for the object
        closures % coordinates of the closure branches
        inductors % list of inductor positions
        modes % list of circuit modes
        truncations % list of truncations associated with each mode
        oplist % list of current and charge operators available for the circuit
        rotation % rotation between the circuit graph node representation and the new modes representation
%     end
%     properties (Access=private)
        ioperators % current operators coordinates
        parammatext % contains the new capacitive, inductive and Josephson matrices, 
        % once the circuit has been coupled to other circuits
    end
    properties(Dependent)
        gatescap % (Diagonal) capacitance matrix associated with the gates capacitors defined when introducing a voltage bias
        parammat % Capacitance, inverse inductance and Josephson energy matrices
        extraparams % Dependent physical properties: Josephson energy and capacitance
    end
    methods
        
        %% Constructor, builds a FluxQubit class object with the parameters
        % specific to the default circuit "name"
        % N.B. Ic are critical currents. All parameters are expressed in
        % S.I.
        function obj=FluxQubit(name)
            if nargin==0
            elseif nargin==1
                obj.parammatext={};
                switch(name)
                    case "SCP"
                        obj.circuitname=name;
                        obj.type="charge";
                        obj.parameters=struct('Ic',5e-9,'Csh',1e-15,'Sc',0);
                        obj.fluxb={};
                        obj.voltb={};
                        obj.closures={};
                        obj.inductors={};
                        obj.modes="Jos";
                        obj.truncations=25;
                        obj.oplist={'Q1'};
                        obj.rotation=1;
                        obj.ioperators={};
                        obj.setvoltb(1,[1e-15,0.5])
                    case "fluxRF"
                        obj.circuitname=name;
                        obj.type="flux";
                        obj.parameters=struct('Lq',4.5e-9,'Ic',2e-7,'Csh',4.5e-14);
                        obj.fluxb=struct('fz',0);
                        obj.voltb={};
                        obj.closures=struct('positions',{[1,1]},'names',{'fz'});
                        obj.inductors=struct('positions',{[1,1]},'names',{'Lq'});
                        obj.modes="HO";
                        obj.truncations=50;
                        obj.oplist={'Iz','Q1'};
                        obj.rotation=1;
                        %obj.ioperators=struct('Iz',{{{1,0},(@(x) -x)}});
                        obj.ioperators=struct('Iz',{{1,0}});
                    case "flux3JJ"
                        obj.circuitname=name;
                        obj.type="flux";
                        obj.parameters=struct('Ic',5.1e-7,'alpha',0.8,'Csh',2e-14);
                        obj.fluxb=struct('fz',0);
                        obj.voltb={};
                        obj.closures=struct('positions',{[1,2]},'names',{'fz'});
                        obj.modes=["Jos","Jos"];
                        obj.truncations=[10 10];
                        obj.oplist={'Iz','Q1','Q2'};
                        obj.rotation=eye(2);
                        obj.ioperators=struct('Iz',{{1,0}});
                    case "flux3JJfloat"
                        obj.circuitname=name;
                        obj.type="flux";
                        obj.parameters=struct('Ic',2.1e-7,'alpha',3/7,'Csh',1.3e-14,'Cl',4.5e-14,'Cr',4.5e-14);
                        obj.fluxb=struct('fz',0);
                        obj.voltb={};
                        obj.closures=struct('positions',{[1,2]},'names',{'fz'});
                        obj.modes=["Jos","Jos","Island"];
                        obj.truncations=[5 5 3];
                        obj.oplist={'Iz','Q1','Q2','Q3'};
                        obj.rotation=[[-1,1,0];[-1,0,1];[0.5,0.5,0]];
                        obj.ioperators=struct('Iz',{{3,1}});
                    case "flux3JJLfloat"
                        obj.circuitname=name;
                        obj.type="flux";
                        obj.parameters=struct('Ic',2.1e-7,'alpha',3/7,'Lq',1e-10,'Csh',1.3e-14,'Cl',4.5e-14,'Cr',4.5e-14);
                        obj.fluxb=struct('fz',0);
                        obj.voltb={};
                        obj.closures=struct('positions',{[1,2]},'names',{'fz'});
                        obj.modes=["HO","Jos","Jos","Island"];
                        obj.truncations=[10 5 5 3];
                        obj.oplist={'Iz','Q1','Q2','Q3'};
                        obj.rotation=[[0,0,1,-1];[0,1,-0.5,-0.5];[1,0,-0.5,-0.5];[0.5,0.5,0,0]];
                        obj.ioperators=struct('Iz',{{4,3}});
                    case "flux4JJfloat"
                        obj.circuitname=name;
                        obj.type="flux";
                        obj.parameters=struct('Ic',5.1e-7,'alpha',0.8,'Csh',2e-14);
                        obj.fluxb=struct('fz',0);
                        obj.voltb={};
                        obj.closures=struct('positions',{[1,2]},'names',{'fz'});
                        obj.modes=["Jos","Jos"];
                        obj.truncations=[10 10];
                        obj.oplist={'Iz','Q1','Q2'};
                        obj.rotation=eye(2);
                    case "fluxRF2loops"
                        obj.circuitname=name;
                        obj.type="flux";
                        obj.parameters=struct('Lc',1.2e-9,'Lco1',1e-11,'Lco2',1e-11,'Ic1',1.4e-7,'Ic2',1.4e-7,'Csh',2e-14);
                        obj.fluxb=struct('fz',0,'fx',0);
                        obj.voltb={};
                        %obj.closures=struct('positions',{[1 2],[1 3]},'names',{'f12','f13'}); %f12=fz, f13=fz+fx
                        obj.closures=struct('positions',{[1 2],[1 3]},'names',{{'fz','fx',(@(p) p(1)-p(2)./2)},{'fz','fx',(@(p) p(1)+p(2)./2)}});
                        obj.inductors=struct('positions',{[1,1],[2,2],[3,3]},'names',{'Lc','Lco1','Lco2'});
                        obj.modes=["HO","HO","HO"];
                        obj.truncations=[30, 5, 5];
                        obj.oplist={'Iz','Ix','Q1','Q2','Q3'};
                        obj.rotation=[[1,-0.5,-0.5];[0,0.5,0.5];[0,-1,1]];
                        obj.ioperators=struct('Iz',{{1,0}},'Ix',{{{2,0},{3,0},(@(x,y) -(x-y)./2)}});
                    case "flux3JJL"
                        obj.circuitname=name;
                        obj.type="flux";
                        obj.parameters=struct('Ic',5.1e-7,'alpha',0.8,'Csh',2e-14,'Lq',2e-10);
                        obj.fluxb=struct('fz',0);
                        obj.voltb={};
                        obj.closures=struct('positions',{[2,3]},'names',{'fz'});
                        obj.inductors=struct('positions',{[1,1]},'names',{'Lq'});
                        obj.modes=["HO","Jos","Jos"];
                        obj.truncations=[7 7 7];
                        obj.oplist={'Iz','Q1','Q2','Q3'};
                        obj.rotation=eye(3);
                        % obj.rotation=[[0,1,-1];[1/2,-1/2,-1/2];[1,0,1/2]];
                        % 3 HO modes
                        %obj.ioperators=struct('Iz',{{{1,0},(@(x) -x)}});
                        obj.ioperators=struct('Iz',{{1,0}});
                    case "flux4JJL"
                        obj.circuitname=name;
                        obj.type="flux";
                        obj.parameters=struct('Ic',0.22098e-6,'alpha',0.92316,'Csh',4e-14,'Lq',2.5e-10,'Lco1',1e-11,'Lco2',1e-11);
                        obj.fluxb=struct('fz',0,'fx',0);
                        obj.voltb={};
                        %obj.closures=struct('positions',{[1 4],[1 5]},'names',{'f14','f15'}); %f12=fz, f13=fz+fx
                        obj.closures=struct('positions',{[1 4],[1 5]},'names',{{'fz','fx',(@(p) p(1)-p(2)./2)},{'fz','fx',(@(p) p(1)+p(2)./2)}});
                        obj.inductors=struct('positions',{[2,2],[3,4],[3,5]},'names',{'Lq','Lco1','Lco2'});
                        obj.modes=["Jos","Jos","HO","HO","HO"];
                        obj.truncations=[5 5 3 3 3];
                        obj.oplist={'Iz','Ix','Q1','Q2','Q3','Q4','Q5'};
                        obj.rotation=[[0,0,-1/3,-1/3,-1/3];[1,0,0,0,0];[0,1,0,0,0];[0,0,0,-1,1];[0,0,-1,0.5,0.5]];
                        %obj.ioperators=struct('Iz',{{{2,0},@(x) -x}},'Ix',{{{4,3},{5,3},(@(x,y) -(x-y)./2)}});
                        obj.ioperators=struct('Iz',{{2,0}},'Ix',{{{4,3},{5,3},(@(x,y) -(x-y)./2)}});
                    case "JPSQRF"
                        obj.circuitname=name;
                        obj.type="flux";
                        obj.parameters=struct('Icl',1.2e-7,'Icr',1.2e-7,'Cl',1e-15,'Cr',1e-15,'Csh',1e-15,'Ll',7.5e-10,'Lr',7.5e-10);
                        obj.fluxb=struct('fz',0);
                        obj.voltb={};
                        obj.closures=struct('positions',{[1,2]},'names',{'fz'});
                        obj.inductors=struct('positions',{[1,1],[3,3]},'names',{'Ll','Lr'});
                        obj.modes=["HO","HO","Jos"];
                        obj.truncations=[5,4,8];
                        obj.oplist={'Iz','Q1','Q2','Q3'};
                        obj.rotation=[[1,0,-1];[1/2,0,1/2];[0,1,0]];
                        %obj.ioperators=struct('Iz',{{{3,0},(@(x) -x)}});
                        obj.ioperators=struct('Iz',{{1,0}});
                        obj.setvoltb(2,[5e-15,0.5])
                    case "JPSQRF2loops"
                        obj.circuitname=name;
                        obj.type="flux";
                        obj.parameters=struct('IclT',1.32e-7,'IclB',1.32e-7,'IcrT',1.32e-7,'IcrB',1.32e-7,'Cl',0.1e-15,...
                            'Cr',0.1e-15,'Csh',0,'Cshl',0,'Cshr',0,'Ll',7.5e-10,'Lr',7.5e-10,'LlT',5e-11,'LlB',5e-11,'LrT',5e-11,'LrB',5e-11);
                        obj.fluxb=struct('fz',0,'fL',0,'fR',0);
                        obj.voltb={};
                        %obj.closures=struct('positions',{[2,3],[2,4],[2,5]},'names',{'f23','f24','f25'});
                        obj.closures=struct('positions',{[2,3],[2,4],[2,5]},'names',{{'fz','fL','fR',(@(p) -p(1)+(p(2)+p(3))./2)},{'fz','fL','fR',(@(p) -p(1)+(p(3)-p(2))./2)},'fR'});
                        obj.inductors=struct('positions',{[1,1],[7,7],[1,3],[1,4],[5,7],[6,7]},'names',{'Ll','Lr','LlT','LlB','LrT','LrB'});
                        obj.modes=["HO","HO","HO","HO","HO","HO","Jos"];
                        obj.truncations=[2,6,3,2,2,4,5];
                        obj.oplist={'Iz','Il','Ir','Q1','Q2','Q3','Q4','Q5','Q6','Q7'};
                        obj.rotation=[[1/4,0,-1/8,-1/8,1/8,1/8,-1/4];[0,0,1/4,1/4,1/4,1/4,0];[0,0,1/2,1/2,-1/2,-1/2,0];...
                            [0,0,1,-1,0,0,0];[0,0,0,0,-1,1,0];[1/2,0,0,0,0,0,1/2];[0,1,-1/4,-1/4,-1/4,-1/4,0]];
%                         obj.rotation=[[0,0,-1/2,0,0,-1/2,0];[0,0,0,0,0,0,1];[0,0,1/2,0,0,-1/2,0];...
%                             [1,0,0,0,0,0,0];[0,0,0,0,1,0,0];[0,0,0,1,0,0,0];[0,1,0,0,0,0,0]];
                        obj.ioperators=struct('Iz',{{1,0}},'Il',{{{3,1},{4,1},(@(x,y) -(x-y)./2)}},'Ir',{{{5,7},{6,7},(@(x,y) -(x-y)./2)}});
                        %obj.ioperators=struct('Iz',{{{3,0},(@(x) -x)}});
                        obj.setvoltb(2,[5e-15,0.5])
                    case "tuneC"
                        obj.circuitname=name;
                        obj.type="";
                        obj.parameters=struct('C',5e-15,'Ic1',1.2e-7,'Ic2',1.2e-7);
                        obj.fluxb={};
                        obj.voltb={};
                        obj.closures={};
                        obj.inductors={};
                        obj.modes=["Island","Jos","Jos"];
                        obj.truncations=[5,7,7];
                        obj.oplist={'Iz','Q1','Q2','Q3'};
                        obj.rotation=eye(3);
                        %obj.ioperators=struct('Iz',{{{3,0},(@(x) -x)}});
                        obj.ioperators={};
                        obj.setvoltb(1,[5e-15,0.5])
                    case "LCres"
                        obj.circuitname=name;
                        obj.type="";
                        obj.parameters=struct('C',7.8e-13,'L',1.95e-9);
                        obj.fluxb={};
                        obj.voltb={};
                        obj.closures={};
                        obj.inductors=struct('positions',{[1,1]},'names',{'L'});
                        obj.modes="HO";
                        obj.truncations=20;
                        obj.oplist={'Q1','I'};
                        obj.rotation=1;
                        obj.ioperators=struct('I',{{1,0}});
                    case "squidres"
                        obj.circuitname=name;
                        obj.type="";
                        obj.parameters=struct('Lc',1.37e-10,'Lco',1e-11,'Ic1',1e-6,'Ic2',1e-6,'Csh',1e-15,'wr',7,'Zr',50);
                        obj.fluxb=struct('fz',0,'fx',0);
                        obj.voltb={};
                        obj.closures=struct('positions',{[1 2],[1 3]},'names',{'f12','f13'}); %f12=fz, f13=fz+fx
                        %obj.closures=struct('positions',{[1 2],[1 3]},'names',{{'fz','fx',(@(p) p(1)-p(2)./2)},{'fz','fx',(@(p) p(1)+p(2)./2)}});
                        obj.modes=["HO","HO","HO","HO"];
                        obj.truncations=[7 7 5 5];
                        obj.oplist={'Iz','Ix','Q1','Q2','Q3','Q4'};
                        obj.rotation=[[1,0,0,0];[-1,0,0,1];[0,1,0,0];[0,0,1,0]];
                        obj.ioperators=struct('Iz',{{{1,0},(@(x) -x)}},'Ix',{{{2,0},{3,0},(@(x,y) -(x-y)./2)}});
                    case "capacitor"
                        obj.circuitname=name;
                        obj.type="";
                        obj.parameters=struct('C',1e-13);
                        obj.fluxb={};
                        obj.voltb={};
                        obj.closures={};
                        obj.inductors={};
                        obj.modes="Island";
                        obj.truncations=20;
                        obj.oplist={'Q1'};
                        obj.rotation=1;
                        obj.ioperators={};
                    case "SCPcoupler"
                        obj.circuitname=name;
                        obj.type="";
                        obj.parameters=struct('Ic',5e-9,'C0',1e-15,'Csh',1e-15,'Sc',0);
                        obj.fluxb={};
                        obj.voltb={};
                        obj.closures={};
                        obj.inductors={};
                        obj.modes=["Jos","Island"];
                        obj.truncations=[10,5];
                        obj.oplist={'Q1'};
                        obj.rotation=[[1,-1];[1,1]];
                        obj.ioperators={};
                        obj.setvoltb(2,[1e-15,0.5])
                    otherwise
                        fprintf('Use a valid circuit name\n');
                end
            end
        end
        
        %% Calculates internal capacitances and Josephson Energies (from Ic)
        function a=get.extraparams(obj)
            if isfield(obj.parameters,'Sc') % Use standard values if current density and spacific capacitance are not specified
                Sc0=obj.parameters.Sc;
            else
                Sc0=obj.Sc;
            end
            if isfield(obj.parameters,'Jc')
                Jc0=obj.parameters.Jc;
            else
                Jc0=obj.Jc;
            end
            if obj.circuitname=="SCP"
                a=struct('CJ',Sc0/Jc0*obj.parameters.Ic*10^(-9),'EJ',obj.h*obj.parameters.Ic/(4*pi*obj.e));
            elseif obj.circuitname=="SCPcoupler"
                a=struct('CJ',Sc0/Jc0*obj.parameters.Ic*10^(-9),'EJ',obj.h*obj.parameters.Ic/(4*pi*obj.e));
            elseif obj.circuitname=="fluxRF"
                a=struct('CJ',Sc0/Jc0*obj.parameters.Ic*10^(-9),'EJ',obj.h*obj.parameters.Ic/(4*pi*obj.e),...
                    'betaL',2*pi*obj.parameters.Ic*obj.parameters.Lq/obj.Phi0,'meff',(obj.Phi0/(2*pi))^2*(obj.parameters.Csh+...
                    Sc0/Jc0*obj.parameters.Ic*10^(-9)));
            elseif obj.circuitname=="flux3JJ"
                a=struct('CJ',Sc0/Jc0*obj.parameters.Ic*10^(-9),'CJa',...
                    Sc0/Jc0*obj.parameters.Ic*10^(-9)*obj.parameters.alpha,...
                    'EJ',obj.h*obj.parameters.Ic/(4*pi*obj.e),'EJa',...
                    obj.h*obj.parameters.Ic/(4*pi*obj.e)*obj.parameters.alpha);
            elseif obj.circuitname=="flux3JJfloat"
                a=struct('CJ',Sc0/Jc0*obj.parameters.Ic*10^(-9),'CJa',...
                    Sc0/Jc0*obj.parameters.Ic*10^(-9)*obj.parameters.alpha,...
                    'EJ',obj.h*obj.parameters.Ic/(4*pi*obj.e),'EJa',...
                    obj.h*obj.parameters.Ic/(4*pi*obj.e)*obj.parameters.alpha);
            elseif obj.circuitname=="flux3JJLfloat"
                a=struct('CJ',Sc0/Jc0*obj.parameters.Ic*10^(-9),'CJa',...
                    Sc0/Jc0*obj.parameters.Ic*10^(-9)*obj.parameters.alpha,...
                    'EJ',obj.h*obj.parameters.Ic/(4*pi*obj.e),'EJa',...
                    obj.h*obj.parameters.Ic/(4*pi*obj.e)*obj.parameters.alpha);
            elseif obj.circuitname=="fluxRF2loops"
                a=struct('CJ1',Sc0/Jc0*obj.parameters.Ic1*10^(-9),'CJ2',Sc0/Jc0*obj.parameters.Ic1*10^(-9),...
                    'betaL',2*pi*(obj.parameters.Ic1+obj.parameters.Ic2)*obj.parameters.Lc/obj.Phi0,'EJ1',obj.h*obj.parameters.Ic1/(4*pi*obj.e),'EJ2',obj.h*obj.parameters.Ic2/(4*pi*obj.e));
            elseif obj.circuitname=="flux3JJL"
                a=struct('CJ',Sc0/Jc0*obj.parameters.Ic*10^(-9),'CJa',...
                    Sc0/Jc0*obj.parameters.Ic*10^(-9)*obj.parameters.alpha,...
                    'EJ',obj.h*obj.parameters.Ic/(4*pi*obj.e),'EJa',...
                    obj.h*obj.parameters.Ic/(4*pi*obj.e)*obj.parameters.alpha);
            elseif obj.circuitname=="flux4JJL"
                a=struct('CJ',Sc0/Jc0*obj.parameters.Ic*10^(-9),'CJa',...
                    Sc0/Jc0*obj.parameters.Ic*10^(-9)*obj.parameters.alpha,...
                    'EJ',obj.h*obj.parameters.Ic/(4*pi*obj.e),'EJa',...
                    obj.h*obj.parameters.Ic/(4*pi*obj.e)*obj.parameters.alpha);
            elseif obj.circuitname=="JPSQRF"
                a=struct('CJl',Sc0/Jc0*obj.parameters.Icl*10^(-9),'CJr',Sc0/Jc0*obj.parameters.Icr*10^(-9),...
                    'EJl',obj.h*obj.parameters.Icl/(4*pi*obj.e),'EJr',...
                    obj.h*obj.parameters.Icr/(4*pi*obj.e));
            elseif obj.circuitname=="JPSQRF2loops"
                a=struct('CJlT',Sc0/Jc0*obj.parameters.IclT*10^(-9),'CJlB',Sc0/Jc0*obj.parameters.IclB*10^(-9),'CJrT',Sc0/Jc0*obj.parameters.IcrT*10^(-9),...
                    'CJrB',Sc0/Jc0*obj.parameters.IcrB*10^(-9),'EJlT',obj.h*obj.parameters.IclT/(4*pi*obj.e),'EJlB',obj.h*obj.parameters.IclB/(4*pi*obj.e),'EJrT',...
                    obj.h*obj.parameters.IcrT/(4*pi*obj.e),'EJrB',obj.h*obj.parameters.IcrB/(4*pi*obj.e));
            elseif obj.circuitname=="tuneC"
                a=struct('CJ1',Sc0/Jc0*obj.parameters.Ic1*10^(-9),'EJ1',obj.h*obj.parameters.Ic1/(4*pi*obj.e),'CJ2',...
                    Sc0/Jc0*obj.parameters.Ic2*10^(-9),'EJ2',obj.h*obj.parameters.Ic2/(4*pi*obj.e));
            elseif obj.circuitname=="LCres"
                a=struct('omegap',1/sqrt(obj.parameters.C*obj.parameters.L)/(2*pi)*1e-9,'Z',sqrt(obj.parameters.L/obj.parameters.C));
            elseif obj.circuitname=="squidres"
                a=struct('CJ1',Sc0/Jc0*obj.parameters.Ic1*10^(-9),'CJ2',Sc0/Jc0*obj.parameters.Ic1*10^(-9),...
                    'EJ1',obj.h*obj.parameters.Ic1/(4*pi*obj.e),'EJ2',obj.h*obj.parameters.Ic2/(4*pi*obj.e),...
                    'Lr',obj.parameters.Zr/(2*pi*1e9*obj.parameters.wr),'Cr',(obj.parameters.Zr*2*pi*1e9*obj.parameters.wr)^-1);
            else
                a={};
            end
        end
        
        %% Automatically calculates Cmat, Lmat and Ejmat starting from the circuit parameters...
        % (Cmat has the sum of the capacitances connected to node i in the 
        % position (i,i) and minus the sum of the capacitances connecting 
        % the nodes i and j in the position (i,j). Lmat has the same 
        % structure, but with inverse inductances instead of capacitances. 
        % Ejmat is a triangular matrix and has in the position (i,i) the  
        % Josephson energy of the junction connecting the node i to the 
        % ground and in the position (i,j) the one of the junction 
        % connecting nodes i and j

        function a=get.parammat(obj)
            if isempty(obj.parammatext)
                switch(obj.circuitname)
                    case "SCP"
                        a=struct('Cmat',obj.parameters.Csh+obj.extraparams.CJ,'Lmat',0,...
                            'Ejmat',obj.extraparams.EJ);
                    case "SCPcoupler"
                        a=struct('Cmat',[[obj.parameters.Csh+obj.extraparams.CJ+obj.parameters.C0,...
                            -obj.parameters.Csh-obj.extraparams.CJ];[-obj.parameters.Csh-obj.extraparams.CJ,...
                            obj.parameters.Csh+obj.extraparams.CJ]],'Lmat',zeros(2),...
                            'Ejmat',[[0,obj.extraparams.EJ];[0,0]]);
                    case "fluxRF"
                        a=struct('Cmat',obj.parameters.Csh+obj.extraparams.CJ,'Lmat',1/obj.parameters.Lq,...
                            'Ejmat',obj.extraparams.EJ);
                    case "flux3JJ"
                        a=struct('Cmat',[[obj.parameters.Csh+obj.extraparams.CJ+obj.extraparams.CJa,...
                            -obj.parameters.Csh-obj.extraparams.CJa];[-obj.parameters.Csh-obj.extraparams.CJa,...
                            obj.parameters.Csh+obj.extraparams.CJ+obj.extraparams.CJa]],...
                            'Lmat',zeros(2,2),'Ejmat',[[obj.extraparams.EJ,...
                            obj.extraparams.EJa];[0,obj.extraparams.EJ]]);
                    case "fluxRF2loops"
                        a=struct('Cmat',[[obj.parameters.Csh+obj.extraparams.CJ1+obj.extraparams.CJ2,...
                            -obj.extraparams.CJ2,-obj.extraparams.CJ1];[-obj.extraparams.CJ2,obj.extraparams.CJ2,0];...
                            [-obj.extraparams.CJ1,0,obj.extraparams.CJ1]],...
                            'Lmat',diag([1/obj.parameters.Lc,1/obj.parameters.Lco1,1/obj.parameters.Lco2]),'Ejmat',...
                            [[0,obj.extraparams.EJ2,obj.extraparams.EJ1];...
                            [0,0,0];[0,0,0]]);
                    case "flux3JJfloat"
                        a=struct('Cmat',[[obj.extraparams.CJ+obj.extraparams.CJa+obj.parameters.Csh+obj.parameters.Cl,...
                            -obj.parameters.Csh-obj.extraparams.CJa,-obj.extraparams.CJ];[-obj.parameters.Csh-obj.extraparams.CJa,obj.extraparams.CJ+...
                            obj.extraparams.CJa+obj.parameters.Csh+obj.parameters.Cr,...
                            -obj.extraparams.CJ];[-obj.extraparams.CJ,...
                            -obj.extraparams.CJ,2*obj.extraparams.CJ]],...
                            'Lmat',zeros(3),'Ejmat',[[0,obj.extraparams.EJa,obj.extraparams.EJ];...
                            [0,0,obj.extraparams.EJ];[0,0,0]]);
                    case "flux3JJLfloat"
                        a=struct('Cmat',[[obj.extraparams.CJ+obj.extraparams.CJa+obj.parameters.Csh+obj.parameters.Cl,...
                            -obj.parameters.Csh-obj.extraparams.CJa,0,-obj.extraparams.CJ];[-obj.parameters.Csh-obj.extraparams.CJa,obj.extraparams.CJ+...
                            obj.extraparams.CJa+obj.parameters.Csh+obj.parameters.Cr,...
                            -obj.extraparams.CJ,0];[0,-obj.extraparams.CJ,...
                            obj.extraparams.CJ,0];[-obj.extraparams.CJ,0,0,obj.extraparams.CJ]],...
                            'Lmat',[[0,0,0,0];[0,0,0,0];[0,0,1/obj.parameters.Lq,...
                            -1/obj.parameters.Lq];[0,0,-1/obj.parameters.Lq,1/obj.parameters.Lq]],...
                            'Ejmat',[[0,obj.extraparams.EJa,0,obj.extraparams.EJ];...
                            [0,0,obj.extraparams.EJ,0];[0,0,0,0];[0,0,0,0]]);
                    case "flux3JJL"
                        a=struct('Cmat',[[obj.extraparams.CJ,0,-obj.extraparams.CJ];...
                            [0,obj.extraparams.CJ+obj.extraparams.CJa+obj.parameters.Csh,...
                            -obj.extraparams.CJa-obj.parameters.Csh];[-obj.extraparams.CJ,...
                            -obj.extraparams.CJa-obj.parameters.Csh,obj.extraparams.CJ+obj.extraparams.CJa+obj.parameters.Csh]],...
                            'Lmat',diag([1/obj.parameters.Lq,0,0]),'Ejmat',[[0,0,obj.extraparams.EJ];...
                            [0,obj.extraparams.EJ,obj.extraparams.EJa];[0,0,0]]);
                    case "flux4JJL"
                        a=struct('Cmat',[[obj.extraparams.CJ+obj.extraparams.CJa+obj.parameters.Csh,...
                            0,-obj.parameters.Csh,-obj.extraparams.CJa/2,-obj.extraparams.CJa/2];...
                            [0,obj.extraparams.CJ,-obj.extraparams.CJ,0,0];[-obj.parameters.Csh,...
                            -obj.extraparams.CJ,obj.extraparams.CJ+obj.parameters.Csh,0,0];...
                            [-obj.extraparams.CJa/2,0,0,obj.extraparams.CJa/2,0];[-obj.extraparams.CJa/2,0,0,0,obj.extraparams.CJa/2]],...
                            'Lmat',[[0,0,0,0,0];[0,1/obj.parameters.Lq,0,0,0];[0,0,1/obj.parameters.Lco1+1/obj.parameters.Lco2,...
                            -1/obj.parameters.Lco1,-1/obj.parameters.Lco2];[0,0,-1/obj.parameters.Lco1,1/obj.parameters.Lco1,0];...
                            [0,0,-1/obj.parameters.Lco2,0,1/obj.parameters.Lco2]],'Ejmat',[[obj.extraparams.EJ,0,0,...
                            obj.extraparams.EJa/2,obj.extraparams.EJa/2];[0,0,obj.extraparams.EJ,0,0];[0,0,0,0,0];[0,0,0,0,0];[0,0,0,0,0]]);
                    case "JPSQRF"
                        a=struct('Cmat',[[obj.parameters.Cl+obj.extraparams.CJl+obj.parameters.Csh,-obj.extraparams.CJl,-obj.parameters.Csh];...
                            [-obj.extraparams.CJl,obj.extraparams.CJl+obj.extraparams.CJr,...
                            -obj.extraparams.CJr];[-obj.parameters.Csh,...
                            -obj.extraparams.CJr,obj.parameters.Cr+obj.extraparams.CJr+obj.parameters.Csh]],...
                            'Lmat',diag([1/obj.parameters.Ll,0,1/obj.parameters.Lr]),'Ejmat',[[0,obj.extraparams.EJl,0];...
                            [0,0,obj.extraparams.EJr];[0,0,0]]);
                    case "JPSQRF2loops"
                        a=struct('Cmat',[[obj.parameters.Cl+obj.parameters.Cshl+obj.parameters.Csh,-obj.parameters.Cshl,0,0,0,0,-obj.parameters.Csh];...
                            [-obj.parameters.Cshl,obj.parameters.Cshl+obj.parameters.Cshr+obj.extraparams.CJlT+obj.extraparams.CJlB+obj.extraparams.CJrT+obj.extraparams.CJrB,...
                            -obj.extraparams.CJlT,-obj.extraparams.CJlB,-obj.extraparams.CJrT,-obj.extraparams.CJrB,-obj.parameters.Cshr];[0,-obj.extraparams.CJlT,obj.extraparams.CJlT,0,0,0,0];...
                            [0,-obj.extraparams.CJlB,0,obj.extraparams.CJlB,0,0,0];[0,-obj.extraparams.CJrT,0,0,obj.extraparams.CJrT,0,0];[0,-obj.extraparams.CJrB,0,0,0,obj.extraparams.CJrB,0];...
                            [-obj.parameters.Csh,-obj.parameters.Cshr,0,0,0,0,obj.parameters.Cr+obj.parameters.Cshr+obj.parameters.Csh]],...
                            'Lmat',[[1/obj.parameters.Ll+1/obj.parameters.LlT+1/obj.parameters.LlB,0,-1/obj.parameters.LlT,-1/obj.parameters.LlB,0,0,7];[0,0,0,0,0,0,0];...
                            [-1/obj.parameters.LlT,0,1/obj.parameters.LlT,0,0,0,0];[-1/obj.parameters.LlB,0,0,1/obj.parameters.LlB,0,0,0];[0,0,0,0,1/obj.parameters.LrT,0,-1/obj.parameters.LrT];...
                            [0,0,0,0,0,1/obj.parameters.LrB,-1/obj.parameters.LrB];[0,0,0,0,-1/obj.parameters.LrT,-1/obj.parameters.LrB,1/obj.parameters.Lr+1/obj.parameters.LrT+1/obj.parameters.LrB]],...
                            'Ejmat',[[0,0,0,0,0,0,0];[0,0,obj.extraparams.EJlT,obj.extraparams.EJlB,obj.extraparams.EJrT,obj.extraparams.EJrB,0];...
                            [0,0,0,0,0,0,0];[0,0,0,0,0,0,0];[0,0,0,0,0,0,0];[0,0,0,0,0,0,0];[0,0,0,0,0,0,0]]);
                    case "tuneC"
                        a=struct('Cmat',[[obj.parameters.C+obj.extraparams.CJ1+obj.extraparams.CJ2,...
                            -obj.extraparams.CJ1,-obj.extraparams.CJ2];[-obj.extraparams.CJ1,obj.extraparams.CJ1,0];...
                            [-obj.extraparams.CJ2,0,obj.extraparams.CJ2]],...
                            'Lmat',zeros(3),'Ejmat',...
                            [[0,obj.extraparams.EJ1,obj.extraparams.EJ2];...
                            [0,0,0];[0,0,0]]);
                    case "LCres"
                        a=struct('Cmat',obj.parameters.C,'Lmat',1/obj.parameters.L,...
                            'Ejmat',0);
                    case "squidres"
                        a=struct('Cmat',[[obj.parameters.Csh+obj.extraparams.Cr+obj.extraparams.CJ1+obj.extraparams.CJ2,...
                            -obj.extraparams.CJ2,-obj.extraparams.CJ1,-obj.extraparams.Cr];[-obj.extraparams.CJ2,obj.extraparams.CJ2,0,0];...
                            [-obj.extraparams.CJ1,0,obj.extraparams.CJ1,0];[-obj.extraparams.Cr,0,0,obj.extraparams.Cr]],...
                            'Lmat',diag([1/obj.parameters.Lc,2/obj.parameters.Lco,2/obj.parameters.Lco,1/obj.extraparams.Lr])+...
                            [[0,0,0,-1/obj.extraparams.Lr];[0,0,0,0];[0,0,0,0];[-1/obj.extraparams.Lr,0,0,0]],'Ejmat',...
                            [[0,obj.extraparams.EJ2,obj.extraparams.EJ1,0];[0,0,0,0];[0,0,0,0];[0,0,0,0]]);
                    case "capacitor"
                        a=struct('Cmat',obj.parameters.C,'Lmat',0,'Ejmat',0);
                end
                a.Cmat=a.Cmat+obj.gatescap; %Add gates capacitances
            else
                a=obj.parammatext; %allows the external setting of Cmat and Lmat (eg to include loads)
            end
        end
        
        %% Rotation set method: updates the modes according to new R
        function set.rotation(obj,R)
            Rprime=obj.updatemodes(R);
            obj.rotation=Rprime;
        end
        
        %% Set a voltage bias by specifying node number, gate capacitance and gate
        % voltage (in units of electron charge)
        function setvoltb(obj,node,vec)
            if isempty(obj.voltb)
                obj.voltb=struct('node',node,'Cg',vec(1),'Qg',vec(2));
            else
                obj.voltb(end+1)=struct('node',node,'Cg',vec(1),'Qg',vec(2));
            end
        end

        %% Paramsweep method: defines a parameter sweep
        function paramsweep(obj,sweepname,param,range)
            % PARAMSWEEP Defines a parameter sweep.
            %
            %   obj.paramsweep("name",'param',r) defines a sweep with name
            %   "name", which sweeps the parameter 'param' in the range r
            %   (where R is an array).
            if iscell(param)
                toggle=0;
                for paramn=1:length(param)
                    if isfield(obj.parameters,param{paramn}) && isempty(obj.sweeps)
                        obj.sweeps=struct('sweepname',sweepname,'sweeptype','param','parameter',param{paramn},'range',range{paramn});
                    elseif isfield(obj.parameters,param{paramn})
                        if ~sum(cellfun(@(x,y) isequal(x,y),{obj.sweeps.sweepname},repmat({sweepname},1,length(obj.sweeps)))) || toggle
                            obj.sweeps(end+1)=struct('sweepname',sweepname,'sweeptype','param','parameter',param{paramn},'range',range{paramn});
                        else
                            fprintf('Sweep name already in use\n');
                        end
                    else
                        fprintf('Use a valid param sweep parameter\n');
                    end
                    toggle=toggle+1;
                end
            else
                if isfield(obj.parameters,param) && isempty(obj.sweeps)
                    obj.sweeps=struct('sweepname',sweepname,'sweeptype','param','parameter',param,'range',range);
                elseif isfield(obj.parameters,param)
                    if ~sum(cellfun(@(x,y) isequal(x,y),{obj.sweeps.sweepname},repmat({sweepname},1,length(obj.sweeps))))
                        obj.sweeps(end+1)=struct('sweepname',sweepname,'sweeptype','param','parameter',param,'range',range);
                    else
                        fprintf('Sweep name already in use\n');
                    end
                else
                    fprintf('Use a valid param sweep parameter\n');
                end
            end
        end
        
        %% Flux sweep method: defines a flux bias sweep
        function fluxsweep(obj,sweepname,bias,range)
            % FLUXSWEEP Defines a flux bias sweep.
            %
            %   obj.fluxsweep("name",'bias',r) defines a sweep with name
            %   "name", which sweeps the bias flux 'bias' in the range r
            %   (where R is an array).
            if isempty(obj.closures)
                fprintf("This function is only available for circuits with at least one irreducible loop.\n")
            else
                if iscell(bias)
                    toggle=0;
                    for biasn=1:length(bias)
                        if isfield(obj.fluxb,bias{biasn}) && isempty(obj.sweeps)
                            obj.sweeps=struct('sweepname',sweepname,'sweeptype','flux','parameter',bias{biasn},'range',range{biasn});
                        elseif isfield(obj.fluxb,bias{biasn})
                            if ~sum(cellfun(@(x,y) isequal(x,y),{obj.sweeps.sweepname},repmat({sweepname},1,length(obj.sweeps)))) || toggle
                               % all(contains([obj.sweeps.sweepname],sweepname)) || toggle
                                obj.sweeps(end+1)=struct('sweepname',sweepname,'sweeptype','flux','parameter',bias{biasn},'range',range{biasn});
                            else
                                fprintf('Sweep name already in use\n');
                            end
                        else
                            fprintf('Use a valid flux sweep parameter\n');
                        end
                        toggle=toggle+1;
                    end
                else
                    if isfield(obj.fluxb,bias) && isempty(obj.sweeps)
                        obj.sweeps=struct('sweepname',sweepname,'sweeptype','flux','parameter',bias,'range',range);
                    elseif isfield(obj.fluxb,bias)
                        if ~sum(cellfun(@(x,y) isequal(x,y),{obj.sweeps.sweepname},repmat({sweepname},1,length(obj.sweeps))))
                            obj.sweeps(end+1)=struct('sweepname',sweepname,'sweeptype','flux','parameter',bias,'range',range);
                        else
                            fprintf('Sweep name already in use\n');
                        end
                    else
                        fprintf('Use a valid flux sweep parameter\n');
                    end
                end
            end
        end
        
        %% Flux sweep method: defines a flux bias sweep
        function voltsweep(obj,sweepname,bias,range)
            % VOLTSWEEP Defines a voltage bias sweep.
            %
            %   obj.fluxsweep("name",'bias',r) defines a sweep with name
            %   "name", which sweeps the voltage applied to the node 'bias' in the range r
            %   (where R is an array).
            if isempty(obj.voltb)
                fprintf("Set the voltage bias, including gate capacitance, first.\n")
            else
                if iscell(bias)
                    toggle=0;
                    for biasn=1:length(bias)
                        if ~isempty(find([obj.voltb.node]==bias{biasn}, 1)) && isempty(obj.sweeps)
                            obj.sweeps=struct('sweepname',sweepname,'sweeptype','voltage','parameter',bias{biasn},'range',range{biasn});
                        elseif ~isempty(find([obj.voltb.node]==bias{biasn}, 1))
                            if ~sum(cellfun(@(x,y) isequal(x,y),{obj.sweeps.sweepname},repmat({sweepname},1,length(obj.sweeps)))) || toggle
                               % check that the sweep name is not already
                               % in use
                                obj.sweeps(end+1)=struct('sweepname',sweepname,'sweeptype','voltage','parameter',bias{biasn},'range',range{biasn});
                            else
                                fprintf('Sweep name already in use\n');
                            end
                        else
                            fprintf('Define a voltage bias, including a gate capacitance, for this node first.\n');
                        end
                        toggle=toggle+1;
                    end
                else
                    if ~isempty(find([obj.voltb.node]==bias, 1)) && isempty(obj.sweeps)
                        obj.sweeps=struct('sweepname',sweepname,'sweeptype','voltage','parameter',bias,'range',range);
                    elseif ~isempty(find([obj.voltb.node]==bias, 1))
                        if ~sum(cellfun(@(x,y) isequal(x,y),{obj.sweeps.sweepname},repmat({sweepname},1,length(obj.sweeps))))
                            obj.sweeps(end+1)=struct('sweepname',sweepname,'sweeptype','voltage','parameter',bias,'range',range);
                        else
                            fprintf('Sweep name already in use\n');
                        end
                    else
                        fprintf('Define a voltage bias, including a gate capacitance, for this node first.\n');
                    end
                end
                fprintf('Make sure the charge you want to sweep is set to zero in voltb.\n')
            end
        end
        
        
        %% Calculates contributions to the capacitance matrix due to the
        % optional added gate capacitors
        function a=get.gatescap(obj)
            a=zeros(length(obj.modes));
            for ind=1:length(obj.voltb)
                a(obj.voltb(ind).node,obj.voltb(ind).node)=a(obj.voltb(ind).node,obj.voltb(ind).node)+obj.voltb(ind).Cg;
            end
        end
        
        %% Calculates the spectrum of states for a fixed parameter set
        function out=calculateE(obj,levelsout,relative,option)
            % CALCULATEE Calculates the energy spectrum of the system given
            % its current parameters and biases.
            %
            %   out=obj.calculateE(levelsout) produces the first levelsout energy
            %   levels of the system.
            %   out=obj.calculateE(obj,levelsout,'Relative',option) subtracts
            %   the ground state energy from every level if option is set
            %   to true.
            if nargin<2
                fprintf("Specify the number of output levels\n");
            else
                if length(obj.truncations)==length(obj.modes)
                    H=obj.getH;
                    H=sparse(H);
                    if isreal(H)
                        out=eigs(H,levelsout,'SA'); %Avoids MATLAB complaining
                    else
                        out=eigs(H,levelsout,'SR');
                    end
                    out=sort(real(out));
                    if nargin==4 && isequal(relative,'Relative') && option==true
                        out=out-out(1,1);
                    end
                else
                    fprintf("Invalid truncation vector: %i truncation values are required\n",length(obj.modes));
                end
            end
        end
        
        %% Calculates the spectrum as a function of one varying parameter
        % A parameter/flux bias sweep needs to be defined beforehand using
        % the methods paramseep of fluxsweep

        function out=sweepE(obj,sweepnm,levelsout,relative,optionr,plotopt,optionp)
            % SWEEPE Calculates the eigenspectrum corresponding to a
            % parameter sweep.
            %
            %   out=obj.sweepE(sweepnm,levelsout)
            %   produces the first levelsout eigenvalues of the system as a
            %   function of the parameter sweep defined under the name
            %   sweepnm.
            %   out=obj.sweepE(sweepnm,levelsout,'Relative',optionr) subtracts the
            %   ground state energy from every level if optionr is set to true.
            %   out=obj.sweepE(sweepnm,levelsout,'Relative',optionr,'Plot',optionp)
            %   produces a plot of the results if optionp is set tot true
%             if isempty(obj.closures)
%                 fprintf("This function is only available for circuits with at least one irreducible loop.\n")
%             elseif nargin < 3
            if nargin < 3
                fprintf("Specify sweepname and number of output levels\n");
            elseif ~isequal(length(obj.truncations),length(obj.modes))
                fprintf("Invalid truncation vector: %i truncation values are required\n",length(obj.modes));
            else
                if isempty(obj.sweeps)
                    fprintf("Define a sweep first using the function paramsweep or fluxsweep\n");
                else
                    numsweptpar=0; %determine the number of swept parameters
                    for sweepn=1:length(obj.sweeps)
                        if isequal(obj.sweeps(sweepn).sweepname,sweepnm)
                            numsweptpar=numsweptpar+1;
                        end
                    end
                    if numsweptpar==0 %Check that the correct sweep name was called
                        fprintf("Specify an existent sweep name\n");
                    else
                        %copy sweeps
                        sweep=struct('sweepname',sweepnm,'sweeptype',cell(1,numsweptpar),'parameter',...
                            cell(1,numsweptpar),'range',cell(1,numsweptpar));
                        param=1;
                        for sweepn=1:length(obj.sweeps)
                            if isequal(obj.sweeps(sweepn).sweepname,sweepnm)
                                sweep(param)=obj.sweeps(sweepn); % Only copy the sweeps with the right name
                                param=param+1;
                            end
                        end
                        sweeplengths=cellfun(@length,{sweep.range}); %checks that all swept parameters take the same number of values
                        if all(sweeplengths==sweeplengths(1))
                            switch(sweep(1).sweeptype)
                                case 'param' %parameter sweep: requires calculating the full hamiltonian every time
                                    paramold=zeros(1,numsweptpar); %copy the original value of the swept parameter
                                    for paramn=1:numsweptpar
                                        paramold(1,paramn)=obj.parameters.(sweep(paramn).parameter);
                                    end
                                    out=zeros(length(sweep(1).range),levelsout); %Initialise output energy eigenvalues matrix
                                    f=waitbar(1/length(sweep(1).range),strcat('Sweep progress:',num2str(100/length(sweep(1).range),2),'%'));
                                    for iter=1:length(sweep(1).range)
                                        waitbar(iter/length(sweep(1).range),f,strcat('Sweep progress:',num2str(100*iter/length(sweep(1).range),2),'%'));
                                        for paramn=1:numsweptpar %update swept parameters
                                            obj.parameters.(sweep(paramn).parameter)=sweep(paramn).range(iter);
                                        end
                                        H=sparse(obj.getH); %Get circuit Hamiltonian
                                        if isreal(H) %Avoid Matlab warnings
                                            eigenvals=eigs(H,levelsout,'SA'); %Determine eigenvalues
                                        else
                                            eigenvals=eigs(H,levelsout,'SR');
                                        end
                                        out(iter,:)=sort(real(eigenvals));
                                        if nargin>4 && isequal(relative,'Relative') && optionr
                                            out(iter,:)=out(iter,:)-out(iter,1);
                                        end
                                    end
                                    for paramn=1:numsweptpar
                                        obj.parameters.(sweep(paramn).parameter)=paramold(paramn);
                                    end
                                case 'flux' %flux sweep, only the bias matrix needs to be recalculated
                                    %nummodes=length(obj.modes);
                                    EJ=obj.parammat.Ejmat; %Josephonon energies matrix
                                    out=zeros(length(sweep(1).range),levelsout);
                                    HLC=sparse(obj.getHLC); %Linear part of the Hamiltonian
                                    Dmat=obj.getDops; %Displacement operators
                                    f=waitbar(1/length(sweep(1).range),strcat('Sweep progress:',num2str(100/length(sweep(1).range),2),'%'));
                                    for iter=1:length(sweep(1).range)
                                        if iter==1
                                            fluxbcopy=obj.fluxb;
                                        end
                                        waitbar(iter/length(sweep(1).range),f,strcat('Sweep progress:',num2str(100*iter/length(sweep(1).range),2),'%'));
                                        HJ=eye(length(HLC)).*sum(sum(EJ));
                                        for fluxind=1:length(sweep)
                                            fluxbcopy.(sweep(fluxind).parameter)=sweep(fluxind).range(iter);
                                        end
                                        %biasmatrix=obj.getBiasm(sweep,iter); %Update biases from sweep
                                        biasmatrix=obj.getBiasm(fluxbcopy); %Update biases from sweep
                                        for ind=find(EJ)' %Sum over nonzero terms in the EJ matrix
                                            [Dmatrow,Dmatcolumn]=ind2sub(size(EJ,1),ind);
%                                         for Dmatrow=1:nummodes
%                                             for Dmatcolumn=Dmatrow:nummodes
%                                                 if EJ(Dmatrow,Dmatcolumn)~=0 %sum only on the branches containing a JJ
                                            % Josephson energy Hamiltonian
                                            HJ=HJ-EJ(Dmatrow,Dmatcolumn)/2.*(biasmatrix(Dmatrow,Dmatcolumn).*Dmat{Dmatrow,Dmatcolumn}+...
                                                conj(biasmatrix(Dmatrow,Dmatcolumn)).*Dmat{Dmatrow,Dmatcolumn}');
%                                                 end
%                                             end
%                                         end
                                        end
                                        HJ=sparse(HJ./obj.h*1e-9);
                                        if isreal(HLC+HJ) %Avoid Matlab warnings
                                            eigenvals=eigs(HLC+HJ,levelsout,'SA'); %Determine eigenvalues
                                            %eigenvals=eigs((HLC+HJ+HLC'+HJ')./2,levelsout,'SA');
                                            %Determine eigenvalues,
                                            %ensuring hermiticity (doesn't
                                            %seem to change the result)
                                        else
                                            eigenvals=eigs(HLC+HJ,levelsout,'SR');
                                            %eigenvals=eigs((HLC+HJ+HLC'+HJ')./2,levelsout,'SR');
                                        end
                                        out(iter,:)=sort(real(eigenvals));
                                        if nargin>4 && isequal(relative,'Relative') && optionr
                                            out(iter,:)=out(iter,:)-out(iter,1); %subtract ground state
                                        end
                                    end
%                                     if nargin>4 && isequal(relative,'Relative') && ~optionr
%                                         out=out-min(min(out)); %Subtract the minimum of the spectrum
%                                     end
                                case 'voltage'
                                    %voltage sweep, only the terms relative to the interaction with the
                                    %bias charge need to be recalculated;
                                    out=zeros(length(sweep(1).range),levelsout);
                                    HLC=sparse(obj.getHLC(true)); %Linear part of the Hamiltonian
                                    HJ=sparse(obj.getHJ); %Josephson part of the Hamiltonian
                                    Hlen=length(HJ);
                                    R=obj.rotation;
                                    Rlen=length(R);
                                    Cmat=obj.parammat.Cmat;
                                    Cinv=(R/Cmat)*R';
                                    [~,Q]=obj.getPQ();
                                    f=waitbar(1/length(sweep(1).range),strcat('Sweep progress:',num2str(100/length(sweep(1).range),2),'%'));
                                    for iter=1:length(sweep(1).range)
                                        waitbar(iter/length(sweep(1).range),f,strcat('Sweep progress:',num2str(100*iter/length(sweep(1).range),2),'%'));
                                        Hbias=zeros(Hlen);
                                        Qoff=zeros(1,Rlen);
                                        for ind=1:length(sweep)
                                            Qoff(sweep(ind).parameter)=sweep(ind).range(iter)*2*obj.e;
                                            for ind2=1:length(obj.voltb)
                                                if obj.voltb(ind2).node==sweep(ind).parameter
                                                    Qoff(sweep(ind).parameter)=Qoff(sweep(ind).parameter)+obj.voltb(ind2).Qg*2*obj.e;
                                                end
                                            end
                                        end
                                        Qoff=inv(R)'*Qoff';
                                        Qoff=Qoff';
                                        if any(Qoff)
                                            for ind=find(Qoff)
                                                Hbias=Hbias+Qoff(ind).*ncon({Cinv(ind,:)',Q},{1,[-1 -2 1]})+...
                                                    Cinv(ind,ind)*(Qoff(ind))^2/2.*eye(Hlen);
                                            end
                                        end
                                        Hbias=sparse(Hbias./obj.h*1e-9);
                                        if isreal(HLC+HJ+Hbias) %Avoid Matlab warnings
                                            eigenvals=eigs(HLC+HJ+Hbias,levelsout,'SA'); %Determine eigenvalues
                                            %eigenvals=eigs((HLC+HJ+HLC'+HJ')./2,levelsout,'SA');
                                            %Determine eigenvalues,
                                            %ensuring hermiticity (doesn't
                                            %seem to change the result)
                                        else
                                            eigenvals=eigs(HLC+HJ+Hbias,levelsout,'SR');
                                            %eigenvals=eigs((HLC+HJ+HLC'+HJ')./2,levelsout,'SR');
                                        end
                                        out(iter,:)=sort(real(eigenvals));
                                        if nargin>4 && isequal(relative,'Relative') && optionr
                                            out(iter,:)=out(iter,:)-out(iter,1); %subtract ground state
                                        end
                                    end
%                                     if nargin>4 && isequal(relative,'Relative') && ~optionr
%                                         out=out-min(min(out)); %Subtract the minimum of the spectrum
%                                     end
                                    
                            end
                            close(f)
                            if nargin>6 && isequal(plotopt,'Plot') && optionp %Plot results
                                figure
                                plot(sweep(1).range,out,'LineWidth',2);
                                set(gca,'fontsize',18);
                                xlim([min(sweep(1).range),max(sweep(1).range)])
                                if isequal(sweep(1).sweeptype,'voltage')
                                    labelx=strcat('$Q_{g',num2str(sweep(1).parameter),'}(2e)$');
                                else
                                    labelx=strcat('\textit{',sweep(1).parameter,'}');
                                end
                                xlabel(labelx,'FontSize',22,'Interpreter','latex')
                                if isequal(sweep(1).sweeptype,'voltage')
                                    if nargin>4 && isequal(relative,'Relative') && optionr
                                        labely=strcat('$E_i-E_0(Q_{g',num2str(sweep(1).parameter),'})$ (GHz)');
                                    else
                                        labely=strcat('$E_i(Q_{g',num2str(sweep(1).parameter),'})$ (GHz)');
                                    end
                                else
                                    if nargin>4 && isequal(relative,'Relative') && optionr
                                        labely=strcat('$E_i-E_0(',sweep(1).parameter,')$ (GHz)');
                                    else
                                        labely=strcat('$E_i(',sweep(1).parameter,')$ (GHz)');
                                    end
                                end
                                ylabel(labely,'FontSize',22,'Interpreter','latex')
                            end
                        else
                            fprintf("Inconsistent params ranges\n");
                        end
                    end
                end
            end
        end
        
        %% Calculates the expectation value of the current/node charge operator
        % iopname between the states initstate and finstate
        function out=matelem(obj,iopname,initstate,finstate)
            % MATELEM Calculates the matrix element of a current/charge
            % operator.
            %
            %   out=obj.matelem('iopname',initstate,finstate) Calculates the
            %   matrix element of the operator 'iopname' between the energy
            %   eigenstates initstate and finstate. To calculate expectation
            %   values of functions of the operator, Iopname can be replaced
            %   with a cell in which the first item is the operator name
            %   and the second is an anonymous function, i.e. @(x) f(x)
            if iscell(iopname)
                opnamecopies=cell(1,length(obj.oplist));
                opnamecopies(1,:)={iopname{1}};
            else
                opnamecopies=cell(1,length(obj.oplist));
                opnamecopies(1,:)={iopname};
            end
            if sum(cellfun(@(x,y) isequal(x,y),obj.oplist,opnamecopies))
                H=obj.getH;
                if iscell(iopname)
                    func=iopname{2};
                    if isequal(iopname{1}(1),'I')
                        Op=obj.getIops(iopname{1});
                        Cfact=1e9; %convert into nano-Ampere
                    elseif isequal(iopname{1}(1),'Q')
                        [~,Q]=obj.getPQ(false,true);
                        Rinv=inv(obj.rotation);
                        Rvec=Rinv(str2double(iopname{1}(2)),:);
                        Op=ncon({Rvec',Q},{1,[-1 -2 1]});
                        Cmat=obj.parammat.Cmat;
                        Cfact=1e9/Cmat(str2double(iopname{1}(2)),str2double(iopname{1}(2))); %convert into nano-Volt
                    end
                    Op=func(Op);
                    Cfact=func(Cfact);
                else
                    if isequal(iopname(1),'I')
                        Op=obj.getIops(iopname);
                        Cfact=1e9; %convert into nano-Ampere
                    elseif isequal(iopname(1),'Q')
                        [~,Q]=obj.getPQ(false,true);
                        Rinv=inv(obj.rotation);
                        Rvec=Rinv(str2double(iopname(2)),:);
                        Op=ncon({Rvec',Q},{1,[-1 -2 1]});
                        Cmat=obj.parammat.Cmat;
                        Cfact=1e9/Cmat(str2double(iopname(2)),str2double(iopname(2))); %convert into nano-Volt
                    end
                end
                if isreal(H)
                    [evect,evals]=eigs(H,max(initstate,finstate),'SA');
                else
                    [evect,evals]=eigs(H,max(initstate,finstate),'SR');
                end
                [~,order]=sort(real(diag(evals)));
                initvec=evect(:,order(initstate));
                initvec=initvec./initvec(find(abs(initvec)>1e-7,1))*abs(initvec(find(abs(initvec)>1e-7,1)));
                finvec=evect(:,order(finstate));
                finvec=finvec./finvec(find(abs(finvec)>1e-7,1))*abs(finvec(find(abs(finvec)>1e-7,1)));
                out=finvec'*Op*initvec*Cfact;
            else
                fprintf("Invalid/undefined operator\n");
            end
                
        end
        
        %% Calculates the expectation value of the current/node charge operator iopname
        % between the states initstate and finstate as a function of some parameter
        function out=sweepOp(obj,iopname,initstate,finstate,sweepname,plotopt,optionp)
            % SWEEPOP Calculates the matrix element of a current/charge
            % operator as a function of a varying parameter.
            %
            %   out=obj.sweepOp('iopname',initstate,finstate,sweepname) calculates
            %   the matrix elements of the operator 'iopname' between the
            %   energy eigenstates initstate and finstate as a function of a
            %   parameter sweep defined under the name sweepname.
            %   out=obj.sweepOp(obj,iopname,initstate,finstate,sweepname,'Plot',optionp)
            %   draws a plot of the result if optionp is set to true.
%             if isempty(obj.closures)
%                 fprintf("This function is only available for circuits with at least one irreducible loop.\n")
%             else
                if iscell(iopname)
                    opnamecopies=cell(1,length(obj.oplist));
                    opnamecopies(1,:)={iopname{1}};
                else
                    opnamecopies=cell(1,length(obj.oplist));
                    opnamecopies(1,:)={iopname};
                end
                if sum(cellfun(@(x,y) isequal(x,y),obj.oplist,opnamecopies))
                    if isempty(obj.sweeps)
                        fprintf("Define a sweep first using the function paramsweep or fluxsweep\n");
                    else
                        numsweptpar=0;
                        for sweepn=1:length(obj.sweeps)
                            if isequal(obj.sweeps(sweepn).sweepname,sweepname)
                                numsweptpar=numsweptpar+1;
                            end
                        end
                        if numsweptpar==0
                            fprintf("Specify an existent sweep name\n");
                        else
                            sweep=struct('sweepname',sweepname,'sweeptype',cell(1,numsweptpar),'parameter',...
                                cell(1,numsweptpar),'range',cell(1,numsweptpar));
                            param=1;
                            for sweepn=1:length(obj.sweeps)
                                if isequal(obj.sweeps(sweepn).sweepname,sweepname)
                                    sweep(param)=obj.sweeps(sweepn); % Only copy the sweeps with the right name
                                    param=param+1;
                                end
                            end
                            sweeplengths=cellfun(@length,{sweep.range}); %checks that all swept parameters take the same number of values
                            if all(sweeplengths==sweeplengths(1))
                                if length(initstate)==1 && length(finstate)==1
                                    out=zeros(1,length(sweep(1).range));
                                elseif max(length(initstate),length(finstate))>1 && min(length(initstate),length(finstate))==1
                                    out=zeros(max(length(initstate),length(finstate)),length(sweep(1).range));
                                elseif length(initstate)==length(finstate) && length(initstate)>1
                                    out=zeros(length(initstate),length(sweep(1).range));
                                else
                                    fprintf("The number of initial and final states need to be equal, unless one of these two numbers is 1.\n")
                                end
                                switch(sweep(1).sweeptype)
                                    case 'param' %parameter sweep: requires calculating the full hamiltonian every time
                                        paramold=zeros(1,numsweptpar); %copy the original value of the swept parameter
                                        for paramn=1:numsweptpar
                                            paramold(1,paramn)=obj.parameters.(sweep(paramn).parameter);
                                        end
                                        out=zeros(1,length(sweep(1).range));
                                        f=waitbar(1/length(sweep(1).range),strcat('Sweep progress:',num2str(100/length(sweep(1).range),2),'%'));
                                        for iter=1:length(sweep(1).range)
                                            waitbar(iter/length(sweep(1).range),f,strcat('Sweep progress:',num2str(100*iter/length(sweep(1).range),2),'%'));
                                            for paramn=1:numsweptpar
                                                obj.parameters.(sweep(paramn).parameter)=sweep(paramn).range(iter);
                                            end
                                            if iscell(iopname)
                                                func=iopname{2};
                                                if isequal(iopname{1}(1),'I')
                                                    Op=obj.getIops(iopname{1});
                                                    Cfact=1e9; %convert into nano-Ampere
                                                elseif isequal(iopname{1}(1),'Q')
                                                    [~,Q]=obj.getPQ(false,true);
                                                    Rinv=inv(obj.rotation);
                                                    Rvec=Rinv(str2double(iopname{1}(2)),:);
                                                    Op=ncon({Rvec',Q},{1,[-1 -2 1]});
                                                    Cmat=obj.parammat.Cmat;
                                                    Cfact=1/Cmat(str2double(iopname(2)),str2double(iopname(2)))*1e9; %convert into nano-Volt
                                                end
                                                Op=func(Op);
                                                Cfact=func(Cfact);
                                            else
                                                if isequal(iopname(1),'I')
                                                    Op=obj.getIops(iopname);
                                                    Cfact=1e9; %convert into nano-Ampere
                                                elseif isequal(iopname(1),'Q')
                                                    [~,Q]=obj.getPQ(false,true);
                                                    Rinv=inv(obj.rotation);
                                                    Rvec=Rinv(str2double(iopname(2)),:);
                                                    Op=ncon({Rvec',Q},{1,[-1 -2 1]});
                                                    Cmat=obj.parammat.Cmat;
                                                    Cfact=1/Cmat(str2double(iopname(2)),str2double(iopname(2)))*1e9; %convert into nano-Volt
                                                end
                                            end
                                            H=obj.getH;
                                            if isreal(H)
                                                [evect,evals]=eigs(H,max(cat(2,initstate,finstate)),'SA');
                                            else
                                                [evect,evals]=eigs(H,max(cat(2,initstate,finstate)),'SR');
                                            end
                                            [~,order]=sort(real(diag(evals)));
                                            if length(initstate)==1 && length(finstate)==1
                                                initvec=evect(:,order(initstate));
                                                initvec=initvec./initvec(find(abs(initvec)>1e-7,1))*abs(initvec(find(abs(initvec)>1e-7,1)));
                                                finvec=evect(:,order(finstate));
                                                finvec=finvec./finvec(find(abs(finvec)>1e-7,1))*abs(finvec(find(abs(finvec)>1e-7,1)));
                                                if initstate==finstate
                                                    out(iter)=real(finvec'*Op*initvec)*Cfact;
                                                else
                                                    out(iter)=finvec'*Op*initvec*Cfact;
                                                end
                                            elseif length(initstate)>1 && length(finstate)==1
                                                finvec=obj.normeigenvectors(evect(:,order(finstate)));
                                                finvec=finvec./finvec(find(abs(finvec)>1e-7,1))*abs(finvec(find(abs(finvec)>1e-7,1)));
                                                initvec=evect(:,order(initstate));
                                                for inits=1:length(initstate)
                                                    invs=initvec(:,inits);
                                                    invs=invs./invs(find(abs(invs)>1e-7,1))*abs(invs(find(abs(invs)>1e-7,1)));
                                                    out(inits,iter)=finvec'*Op*invs*Cfact;
                                                end
                                            elseif length(finstate)>1 && length(initstate)==1
                                                initvec=obj.normeigenvectors(evect(:,order(initstate)));
                                                initvec=initvec./initvec(find(abs(initvec)>1e-7,1))*abs(initvec(find(abs(initvec)>1e-7,1)));
                                                finvec=evect(:,order(finstate));
                                                for finits=1:length(finstate)
                                                    finv=finvec(:,finits);
                                                    finv=finv./finv(find(abs(finv)>1e-7,1))*abs(finv(find(abs(finv)>1e-7,1)));
                                                    out(finits,iter)=finv'*Op*initvec*Cfact;
                                                end
                                            elseif length(initstate)==length(finstate) && length(initstate)>1
                                                initvec=evect(:,order(initstate));
                                                finvec=evect(:,order(finstate));
                                                for inits=1:length(initstate)
                                                    invs=initvec(:,inits);
                                                    invs=invs./invs(find(abs(invs)>1e-7,1))*abs(invs(find(abs(invs)>1e-7,1)));
                                                    finv=finvec(:,inits);
                                                    finv=finv./finv(find(abs(finv)>1e-7,1))*abs(finv(find(abs(finv)>1e-7,1)));
                                                    out(inits,iter)=finv'*Op*invs*Cfact;
                                                end
                                            end
                                        end
                                        for paramn=1:numsweptpar %reset object properties
                                            obj.parameters.(sweep(paramn).parameter)=paramold(paramn);
                                        end
                                    case 'flux' %flux sweep, only the bias matrix needs to be recalculated
                                        nummodes=length(obj.modes);
                                        EJ=obj.parammat.Ejmat;
                                        out=zeros(1,length(sweep(1).range));
                                        HLC=obj.getHLC;
                                        if iscell(iopname)
                                            func=iopname{2};
                                            if isequal(iopname{1}(1),'I')
                                                Op=obj.getIops(iopname{1});
                                                Cfact=1e9; %convert into nano-Ampere
                                            elseif isequal(iopname{1}(1),'Q')
                                                [~,Q]=obj.getPQ(false,true);
                                                Rinv=inv(obj.rotation);
                                                Rvec=Rinv(str2double(iopname{1}(2)),:);
                                                Op=ncon({Rvec',Q},{1,[-1 -2 1]});
                                                Cmat=obj.parammat.Cmat;
                                                Cfact=1e9/Cmat(str2double(iopname{1}(2)),str2double(iopname{1}(2))); %convert into nano-Volt 
                                            end
                                            Op=func(Op);
                                            Cfact=func(Cfact);
                                        else
                                            if isequal(iopname(1),'I')
                                                Op=obj.getIops(iopname);
                                                Cfact=1e9; %convert into nano-Ampere
                                            elseif isequal(iopname(1),'Q')
                                                [~,Q]=obj.getPQ(false,true);
                                                Rinv=inv(obj.rotation);
                                                Rvec=Rinv(str2double(iopname(2)),:);
                                                Op=ncon({Rvec',Q},{1,[-1 -2 1]});
                                                Cmat=obj.parammat.Cmat;
                                                Cfact=1e9/Cmat(str2double(iopname(2)),str2double(iopname(2))); %convert into nano-Volt
                                            end
                                        end
                                        Dmat=obj.getDops;
                                        f=waitbar(1/length(sweep(1).range),strcat('Sweep progress:',num2str(100/length(sweep(1).range),2),'%'));
                                        for iter=1:length(sweep(1).range)
                                            if iter==1
                                                fluxbcopy=obj.fluxb;
                                            end
                                            waitbar(iter/length(sweep(1).range),f,strcat('Sweep progress:',num2str(100*iter/length(sweep(1).range),2),'%'));
                                            HJ=eye(length(HLC)).*sum(sum(EJ));
                                            for fluxind=1:length(sweep)
                                                fluxbcopy.(sweep(fluxind).parameter)=sweep(fluxind).range(iter);
                                            end
                                            biasmatrix=obj.getBiasm(fluxbcopy);
                                            for Dmatrow=1:nummodes
                                                for Dmatcolumn=Dmatrow:nummodes
                                                    if EJ(Dmatrow,Dmatcolumn)~=0 %sum only on the branches containing a JJ
                                                        HJ=HJ-EJ(Dmatrow,Dmatcolumn)/2.*(biasmatrix(Dmatrow,Dmatcolumn).*Dmat{Dmatrow,Dmatcolumn}+...
                                                            conj(biasmatrix(Dmatrow,Dmatcolumn)).*Dmat{Dmatrow,Dmatcolumn}');
                                                    end
                                                end
                                            end
                                            HJ=HJ./obj.h*1e-9;
                                            if isreal(HLC+HJ)
                                                [evect,evals]=eigs(HLC+HJ,max(cat(2,initstate,finstate)),'SA');
                                            else
                                                [evect,evals]=eigs(HLC+HJ,max(cat(2,initstate,finstate)),'SR');
                                            end
                                            [~,order]=sort(real(diag(evals)));
                                            if length(initstate)==1 && length(finstate)==1
                                                initvec=evect(:,order(initstate));
                                                initvec=initvec./initvec(find(abs(initvec)>1e-7,1))*abs(initvec(find(abs(initvec)>1e-7,1)));
                                                finvec=evect(:,order(finstate));
                                                finvec=finvec./finvec(find(abs(finvec)>1e-7,1))*abs(finvec(find(abs(finvec)>1e-7,1)));
                                                if initstate==finstate
                                                    out(iter)=real(finvec'*Op*initvec)*Cfact;
                                                else
                                                    out(iter)=finvec'*Op*initvec*Cfact;
                                                end
                                            elseif length(initstate)>1 && length(finstate)==1
                                                finvec=obj.normeigenvectors(evect(:,order(finstate)));
                                                finvec=finvec./finvec(find(abs(finvec)>1e-7,1))*abs(finvec(find(abs(finvec)>1e-7,1)));
                                                initvec=evect(:,order(initstate));
                                                for inits=1:length(initstate)
                                                    invs=initvec(:,inits);
                                                    invs=invs./invs(find(abs(invs)>1e-7,1))*abs(invs(find(abs(invs)>1e-7,1)));
                                                    out(inits,iter)=finvec'*Op*invs*Cfact;
                                                end
                                            elseif length(finstate)>1 && length(initstate)==1
                                                initvec=obj.normeigenvectors(evect(:,order(initstate)));
                                                initvec=initvec./initvec(find(abs(initvec)>1e-7,1))*abs(initvec(find(abs(initvec)>1e-7,1)));
                                                finvec=evect(:,order(finstate));
                                                for finits=1:length(finstate)
                                                    finv=finvec(:,finits);
                                                    finv=finv./finv(find(abs(finv)>1e-7,1))*abs(finv(find(abs(finv)>1e-7,1)));
                                                    out(finits,iter)=finv'*Op*initvec*Cfact;
                                                end
                                            elseif length(initstate)==length(finstate) && length(initstate)>1
                                                initvec=evect(:,order(initstate));
                                                finvec=evect(:,order(finstate));
                                                for inits=1:length(initstate)
                                                    invs=initvec(:,inits);
                                                    invs=invs./invs(find(abs(invs)>1e-7,1))*abs(invs(find(abs(invs)>1e-7,1)));
                                                    finv=finvec(:,inits);
                                                    finv=finv./finv(find(abs(finv)>1e-7,1))*abs(finv(find(abs(finv)>1e-7,1)));
                                                    out(inits,iter)=finv'*Op*invs*Cfact;
                                                end
                                            end
                                        end
                                    case 'voltage'
                                        %voltage sweep, only the terms relative to the interaction with the
                                        %bias charge need to be recalculated;
                                        out=zeros(1,length(sweep(1).range));
                                        HLC=sparse(obj.getHLC(true)); %Linear part of the Hamiltonian
                                        HJ=sparse(obj.getHJ); %Josephson part of the Hamiltonian
                                        Hlen=length(HJ);
                                        R=obj.rotation;
                                        Rlen=length(R);
                                        Cmat=obj.parammat.Cmat;
                                        Cinv=(R/Cmat)*R';
                                        [~,Q]=obj.getPQ();
                                        f=waitbar(1/length(sweep(1).range),strcat('Sweep progress:',num2str(100/length(sweep(1).range),2),'%'));
                                        if iscell(iopname)
                                            func=iopname{2};
                                            if isequal(iopname{1}(1),'I')
                                                Op=obj.getIops(iopname{1});
                                                Cfact=1e9; %convert into nano-Ampere
                                            elseif isequal(iopname{1}(1),'Q')
                                                [~,Q]=obj.getPQ(false,true);
                                                Rinv=inv(obj.rotation);
                                                Rvec=Rinv(str2double(iopname{1}(2)),:);
                                                Op=ncon({Rvec',Q},{1,[-1 -2 1]});
                                                Cfact=1e9/Cmat(str2double(iopname{1}(2)),str2double(iopname{1}(2))); %convert into nano-Volt
                                            end
                                            Cfact=func(Cfact);
                                        else
                                            func=@(x) x;
                                            if isequal(iopname(1),'I')
                                                Op=obj.getIops(iopname);
                                                Cfact=1e9; %convert into nano-Ampere
                                            elseif isequal(iopname(1),'Q')
                                                [~,Q]=obj.getPQ(false,true);
                                                Rinv=inv(obj.rotation);
                                                Rvec=Rinv(str2double(iopname(2)),:);
                                                Op=ncon({Rvec',Q},{1,[-1 -2 1]});
                                                Cfact=1e9/Cmat(str2double(iopname(2)),str2double(iopname(2))); %convert into nano-Volt
                                            end
                                        end
                                        for iter=1:length(sweep(1).range)
                                            waitbar(iter/length(sweep(1).range),f,strcat('Sweep progress:',num2str(100*iter/length(sweep(1).range),2),'%'));
                                            Hbias=zeros(Hlen);
                                            Qoff=zeros(1,Rlen);
                                            for ind=1:length(sweep)
                                                Qoff(sweep(ind).parameter)=sweep(ind).range(iter)*2*obj.e;
                                                for ind2=1:length(obj.voltb)
                                                    if obj.voltb(ind2).node==sweep(ind).parameter
                                                        Qoff(sweep(ind).parameter)=Qoff(sweep(ind).parameter)+obj.voltb(ind2).Qg*2*obj.e;
                                                    end
                                                end
                                            end
                                            Qoff=inv(R)'*Qoff';
                                            Qoff=Qoff';
                                            if any(Qoff)
                                                for ind=find(Qoff)
                                                    Hbias=Hbias+Qoff(ind).*ncon({Cinv(ind,:)',Q},{1,[-1 -2 1]})+...
                                                        Cinv(ind,ind)*(Qoff(ind))^2/2.*eye(Hlen);
                                                end
                                            end
                                            Hbias=sparse(Hbias./obj.h*1e-9);
                                            if isreal(HLC+HJ+Hbias)
                                                [evect,evals]=eigs(HLC+HJ+Hbias,max(cat(2,initstate,finstate)),'SA');
                                            else
                                                [evect,evals]=eigs(HLC+HJ+Hbias,max(cat(2,initstate,finstate)),'SR');
                                            end
                                            [~,order]=sort(real(diag(evals)));
                                            if length(initstate)==1 && length(finstate)==1
                                                initvec=evect(:,order(initstate));
                                                initvec=initvec./initvec(find(abs(initvec)>1e-7,1))*abs(initvec(find(abs(initvec)>1e-7,1)));
                                                finvec=evect(:,order(finstate));
                                                finvec=finvec./finvec(find(abs(finvec)>1e-7,1))*abs(finvec(find(abs(finvec)>1e-7,1)));
                                                if initstate==finstate
                                                    out(iter)=real(finvec'*func(Op+sweep(ind).range(iter)*2*obj.e*eye(length(Op)))*initvec)*Cfact;
                                                else
                                                    out(iter)=finvec'*func(Op+sweep(ind).range(iter)*2*obj.e*eye(length(Op)))*initvec*Cfact;
                                                end
                                            elseif length(initstate)>1 && length(finstate)==1
                                                finvec=obj.normeigenvectors(evect(:,order(finstate)));
                                                finvec=finvec./finvec(find(abs(finvec)>1e-7,1))*abs(finvec(find(abs(finvec)>1e-7,1)));
                                                initvec=evect(:,order(initstate));
                                                for inits=1:length(initstate)
                                                    invs=initvec(:,inits);
                                                    invs=invs./invs(find(abs(invs)>1e-7,1))*abs(invs(find(abs(invs)>1e-7,1)));
                                                    out(inits,iter)=finvec'*func(Op+sweep(ind).range(iter)*2*obj.e*eye(length(Op)))*invs*Cfact;
                                                end
                                            elseif length(finstate)>1 && length(initstate)==1
                                                initvec=obj.normeigenvectors(evect(:,order(initstate)));
                                                initvec=initvec./initvec(find(abs(initvec)>1e-7,1))*abs(initvec(find(abs(initvec)>1e-7,1)));
                                                finvec=evect(:,order(finstate));
                                                for finits=1:length(finstate)
                                                    finv=finvec(:,finits);
                                                    finv=finv./finv(find(abs(finv)>1e-7,1))*abs(finv(find(abs(finv)>1e-7,1)));
                                                    out(finits,iter)=finv'*func(Op+sweep(ind).range(iter)*2*obj.e*eye(length(Op)))*initvec*Cfact;
                                                end
                                            elseif length(initstate)==length(finstate) && length(initstate)>1
                                                initvec=evect(:,order(initstate));
                                                finvec=evect(:,order(finstate));
                                                for inits=1:length(initstate)
                                                    invs=initvec(:,inits);
                                                    invs=invs./invs(find(abs(invs)>1e-7,1))*abs(invs(find(abs(invs)>1e-7,1)));
                                                    finv=finvec(:,inits);
                                                    finv=finv./finv(find(abs(finv)>1e-7,1))*abs(finv(find(abs(finv)>1e-7,1)));
                                                    out(inits,iter)=finv'*func(Op+sweep(ind).range(iter)*2*obj.e*eye(length(Op)))*invs*Cfact;
                                                end
                                            end
                                        end                                        
                                end
                                close(f)
                                if nargin>6 && isequal(plotopt,'Plot') && optionp
                                    if iscell(iopname)
                                        iopname=iopname{1};
                                    end
                                    figure
                                    if initstate~=finstate
                                        plot(sweep(1).range,round(abs(out),5),'LineWidth',2);
                                        abssing='|';
                                    else
                                        plot(sweep(1).range,round(real(out),5),'LineWidth',2);
                                        abssing='';
                                    end
                                    xlim([min(sweep(1).range),max(sweep(1).range)]);
                                    set(gca,'fontsize',18);
                                    if isequal(sweep(1).sweeptype,'voltage')
                                        labelx=strcat('$Q_{g',num2str(sweep(1).parameter),'}$');
                                    else
                                        labelx=strcat('\textit{',sweep(1).parameter,'}');
                                    end
                                    xlabel(labelx,'FontSize',22,'Interpreter','latex')
                                    ylab=strcat('$',abssing,'\langle',num2str(initstate),'|');
                                    if isequal(iopname,'Iz')
                                        ylab=strcat(ylab,'I_z','|',num2str(finstate),'\rangle',abssing,'$',' (nA)');
                                    elseif isequal(iopname,'Ix')
                                        ylab=strcat(ylab,'I_x','|',num2str(finstate),'\rangle',abssing,'$',' (nA)');
                                    elseif isequal(iopname(1),'Q')
                                        ylab=strcat(ylab,'Q_',iopname(2),'/C','|',num2str(finstate),'\rangle',abssing,'$',' (nV)');
                                    end
                                    ylabel(ylab,'FontSize',22,'Interpreter','latex')
                                end
                            else
                                fprintf("Inconsistent params ranges\n");
                            end
                        end
                    end    
                else
                    fprintf("Invalid/undefined operator\n");
                end
%             end
        end
        
        %% Extract Ising Hamiltonian coefficient using projection on
        % the Pauli operators
        function out=coeffP(obj,opname,method)
            % COEFFP  Determines the coefficient of the Pauli operator
            % opname in the qubit Hamiltonian. Method defines the method
            % used for the reduction. Methods 1 and 3 are two variants of
            % the same local reduction method and should produce the same
            % result. Method 2 defines the computational states at the
            % symmetry point. The coefficients calculated in this case are
            % mixed in a not clear way.
            out=zeros(1,length(opname));
            if method==1
                if isequal(obj.type,"flux")
                    I=sparse(obj.getIops('Iz'));
                elseif isequal(obj.type,"charge")
                    [~,I]=obj.getPQ(false,true);
                    if length(size(I))==3
                        Rinv=obj.rotation';
                        I=squeeze(ncon({Rinv(1,:),I},{[-1,1],[-2 -3 1]})); %Main qubit island must be on node 1!
                        I=sparse(I); 
                    else
                        I=sparse(I);
                    end
                end
                op=obj.getPaulis({'x','y','z','I'},method);
            elseif method==2
                op=obj.getPaulis(opname,method);
            elseif method==3
                if isequal(obj.type,"flux")
                    I=sparse(obj.getIops('Iz'));
                elseif isequal(obj.type,"charge")
                    [~,I]=obj.getPQ(false,true);
                    if length(size(I))==3
                        Rinv=obj.rotation';
                        I=squeeze(ncon({Rinv(1,:),I},{[-1,1],[-2 -3 1]})); %Main qubit island must be on node 1!
                        I=sparse(I); 
                    else
                        I=sparse(I);
                    end
                end
            end
            H=obj.getH;
            switch method
                case 1
                    [ev,en]=eigs(H,10,'sr');
                    [en,order]=sort(real(diag(en)));
                    ev=ev(:,[order(1) order(2)]);
                    en=diag(en(1:2));
                    Ip=ncon({conj(ev),I,ev},{[1 -1],[1 2],[2,-2]});
                    [U,~]=eig(Ip);
                    Hp=U'*en*U;
                    for ind=1:length(opname)
                        switch opname{ind}
                            case 'x'
                                out(ind)=-sqrt(real(trace(op{1}*Hp))^2+real(trace(op{2}*Hp))^2)./2;
                            case 'y'
                                out(ind)=0;
                            case 'z'
                                out(ind)=real(trace(op{3}*Hp))./2;
                            case 'I'
                                out(ind)=real(trace(op{4}*Hp))./2;
                        end
                    end
                case 2
                    for ind=1:length(opname)
                        out(ind)=real(trace(op{ind}*H))./2;
                    end
                case 3
                    [ev,en]=eigs(H,10,'sr');
                    [~,order]=sort(real(diag(en)));
                    ev=ev(:,[order(1) order(2)]);
                    for ind=1:length(opname)
                        op=obj.getPaulis(opname{ind},3,I,ev);
                        out(ind)=real(trace(op{1}*H))./2;
                    end
            end
        end
        
        %% Calculates the spectrum as a function of one varying parameter
        % A parameter/flux bias sweep needs to be defined beforehand using
        % the methods paramseep of fluxsweep

        function out=sweepcoeffP(obj,sweepnm,opnames,method,plotopt,optionp,addcorr,sweeptime)
            % SWEEPCOEFFP Calculates the coefficient of the operators
            % opnames in the qubit Hamiltonian by projection, as a function
            % of a varying parameter.
            % sweepnm specifies the predefined sweepname. opnames is a cell 
            % containing a list of operator names between 'I','x','y' and 'z'
            % method specifies the method used for the reduction.
            % plotopt='Plot' and optionp=1 trigger the plotting of the
            % result. addcorr=1 adds the non-stoquastic dynamical
            % correction (currently under development) In this case sweeptime
            % defines the temporal duration of the sweep.
            % methods 1 and 3 both use local reduction and should give the
            % same result. method 2 defines the computational states at the
            % symmetry point, but produces coefficients mixed in an unclear
            % way. method 4 uses the standard approach to the perturbative
            % reduction, which ensures zero y field and x field independent
            % of fz.
            
            if nargin < 4
                fprintf("Specify sweepname, a set of Pauli operators and the calculation method\n");
            elseif ~isequal(length(obj.truncations),length(obj.modes))
                fprintf("Invalid truncation vector: %i truncation values are required\n",length(obj.modes));
            else
                if isempty(obj.sweeps)
                    fprintf("Define a sweep first using the function paramsweep or fluxsweep\n");
                else
                    if method==1
                        if isequal(obj.type,"flux")
                            I=sparse(obj.getIops('Iz'));
                        elseif isequal(obj.type,"charge")
                            [~,I]=obj.getPQ(false,true);
                            if length(size(I))==3
                                Rinv=obj.rotation';
                                I=squeeze(ncon({Rinv(1,:),I},{[-1,1],[-2 -3 1]})); %Main qubit island must be on node 1!
                                I=sparse(I); 
                            else
                                I=sparse(I);
                            end
                        end
                        ops=obj.getPaulis({'x','y','z','I'},method);
                    elseif method==2
                        ops=obj.getPaulis(opnames,method);
                    elseif method==3
                        if isequal(obj.type,"flux")
                            I=sparse(obj.getIops('Iz'));
                        elseif isequal(obj.type,"charge")
                            [~,I]=obj.getPQ(false,false);
                            if length(size(I))==3
                                Rinv=obj.rotation';
                                I=squeeze(ncon({Rinv(1,:),I},{[-1,1],[-2 -3 1]})); %Main qubit island must be on node 1!
                                I=sparse(I); 
                            else
                                I=sparse(I);
                            end
                        end
                    elseif method==4
                        if isequal(obj.type,"flux")
                            I=sparse(obj.getIops('Iz'));
                        elseif isequal(obj.type,"charge")
                            [~,I]=obj.getPQ(false,false);
                            if length(size(I))==3
                                Rinv=obj.rotation';
                                I=squeeze(ncon({Rinv(1,:),I},{[-1,1],[-2 -3 1]})); %Main qubit island must be on node 1!
                                I=sparse(I); 
                            else
                                I=sparse(I);
                            end
                        end
                    end
                    numsweptpar=0; %determine the number of swept parameters
                    for sweepn=1:length(obj.sweeps)
                        if isequal(obj.sweeps(sweepn).sweepname,sweepnm)
                            numsweptpar=numsweptpar+1;
                        end
                    end
                    if numsweptpar==0 %Check that the correct sweep name was called
                        fprintf("Specify an existent sweep name\n");
                    else
                        if nargin<7
                            addcorr=false;
                        end
                        %copy sweeps
                        sweep=struct('sweepname',sweepnm,'sweeptype',cell(1,numsweptpar),'parameter',...
                            cell(1,numsweptpar),'range',cell(1,numsweptpar));
                        param=1;
                        for sweepn=1:length(obj.sweeps)
                            if isequal(obj.sweeps(sweepn).sweepname,sweepnm)
                                sweep(param)=obj.sweeps(sweepn); % Only copy the sweeps with the right name
                                param=param+1;
                            end
                        end
                        sweeplengths=cellfun(@length,{sweep.range}); %checks that all swept parameters take the same number of values
                        if all(sweeplengths==sweeplengths(1))
                            switch(sweep(1).sweeptype)
                                case 'param' %parameter sweep: requires calculating the full hamiltonian every time
                                    paramold=zeros(1,numsweptpar); %copy the original value of the swept parameter
                                    for paramn=1:numsweptpar
                                        paramold(1,paramn)=obj.parameters.(sweep(paramn).parameter);
                                    end
                                    out=zeros(length(sweep(1).range),length(opnames)); %Initialise output energy eigenvalues matrix
                                    f=waitbar(1/length(sweep(1).range),strcat('Sweep progress:',num2str(100/length(sweep(1).range),2),'%'));
                                    for iter=1:length(sweep(1).range)
                                        waitbar(iter/length(sweep(1).range),f,strcat('Sweep progress:',num2str(100*iter/length(sweep(1).range),2),'%'));
                                        for paramn=1:numsweptpar %update swept parameters
                                            obj.parameters.(sweep(paramn).parameter)=sweep(paramn).range(iter);
                                        end
                                        H=sparse(obj.getH); %Get circuit Hamiltonian
                                        switch method
                                            case 1
                                                [ev,en]=eigs(H,20,'sr');
                                                [en,order]=sort(real(diag(en)));
                                                ev=sparse(ev(:,[order(1) order(2)]));
                                                en=diag(en(1:2));
                                                Ip=ncon({conj(ev),I,ev},{[1 -1],[1 2],[2,-2]});
                                                [U,~]=eig(full(Ip));
                                                Hp=U'*en*U;
                                                for opind=1:length(opnames)
                                                    if opnames{opind}=='z' && obj.fluxb.fz-floor(obj.fluxb.fz)<0.5
                                                        out(iter,opind)=-abs(real(trace(Hp*ops{3})))/2;
                                                    elseif opnames{opind}=='z' && obj.fluxb.fz-floor(obj.fluxb.fz)>0.5
                                                        out(iter,opind)=abs(real(trace(Hp*ops{3})))/2;
                                                    elseif opnames{opind}=='x'
                                                        out(iter,opind)=-sqrt(real(trace(Hp*ops{1}))^2+real(trace(Hp*ops{2}))^2)/2;
                                                    elseif opnames{opind}=='y'
                                                        out(iter,opind)=0;
                                                    elseif opnames{opind}=='I'
                                                        out(iter,opind)=real(trace(Hp*ops{4}))/2;
                                                    end
                                                end
                                            case 2
                                                for opind=1:length(opnames)
                                                    out(iter,opind)=real(trace(H*ops{opind}))/2;
                                                end
                                            case 3
                                                [ev,en]=eigs(H,20,'sr');
                                                [~,order]=sort(real(diag(en)));
                                                ev=ev(:,[order(1) order(2)]);
                                                ops=obj.getPaulis(opnames,3,I,ev);
                                                for opind=1:length(opnames)
                                                    out(iter,opind)=real(trace(H*ops{opind}))/2;
%                                                     out(iter,opind)=real(trace(Hpr*ops{opind}))/2;
                                                end
                                            case 4
                                                if iter==1
                                                    fzold=obj.fluxb.fz;
                                                    obj.fluxb.fz=0.5;
                                                    H0=obj.getH;
                                                    obj.fluxb.fz=fzold;
                                                    [ev,en]=eigs(H0,20,'sr');
                                                    [en,order]=sort(real(diag(en)));
                                                    minus=ev(:,order(2));
                                                    plus=ev(:,order(1));
                                                    up0=(plus-minus)./sqrt(2);
                                                    down0=(plus+minus)./sqrt(2);
                                                    if up0'*I*up0>0 && down0'*I*down0<0
                                                        up=up0;
                                                        down=down0;
                                                    else
                                                        up=down0;
                                                        down=up0;
                                                    end
                                                    Ip=(up'*I*up-down'*I*down)/2;
                                                    Delta=en(2)-en(1);
                                                    trace0=(en(2)+en(1));
                                                end
                                                for opind=1:length(opnames)
                                                    if opnames{opind}=='z'
                                                    	out(iter,opind)=(fzold-0.5)*Ip*obj.Phi0/obj.h*1e-9;
                                                    elseif opnames{opind}=='y'
                                                        out(iter,opind)=0;
                                                    elseif opnames{opind}=='x'
                                                        out(iter,opind)=-Delta/2;
                                                    elseif opnames{opind}=='I'
                                                        out(iter,opind)=trace0/2;
                                                    end
                                                end
                                        end
                                    end
                                    for paramn=1:numsweptpar
                                        obj.parameters.(sweep(paramn).parameter)=paramold(paramn);
                                    end
                                case 'flux' %flux sweep, only the bias matrix needs to be recalculated
                                    %nummodes=length(obj.modes);
                                    EJ=obj.parammat.Ejmat; %Josephonon energies matrix
                                    out=zeros(length(sweep(1).range),length(opnames));
                                    HLC=sparse(obj.getHLC); %Linear part of the Hamiltonian
                                    Dmat=obj.getDops; %Displacement operators
                                    f=waitbar(1/length(sweep(1).range),strcat('Sweep progress:',num2str(100/length(sweep(1).range),2),'%'));
                                    for iter=1:length(sweep(1).range)
                                        if iter==1
                                            fluxbcopy=obj.fluxb;
                                            h0toggle=false;
                                        end
                                        waitbar(iter/length(sweep(1).range),f,strcat('Sweep progress:',num2str(100*iter/length(sweep(1).range),2),'%'));
                                        HJ=eye(length(HLC)).*sum(sum(EJ));
                                        for fluxind=1:length(sweep)
                                            fluxbcopy.(sweep(fluxind).parameter)=sweep(fluxind).range(iter);
                                            if method==4
                                                if sweep(fluxind).parameter=='fz'
                                                    fzcopy=sweep(fluxind).range(iter);
                                                else
                                                    fzcopy=fluxbcopy.fz;
                                                    h0toggle=true;
                                                end
                                            end
                                        end
                                        if h0toggle || (iter==1 && method==4)
                                            HJ0=eye(length(HLC)).*sum(sum(EJ));
                                            fluxbcopy0=fluxbcopy;
                                            fluxbcopy0.fz=0.5;
                                            biasmatrix0=obj.getBiasm(fluxbcopy0); %Update biases from sweep
                                            for ind=find(EJ)' %Sum over nonzero terms in the EJ matrix
                                                [Dmatrow,Dmatcolumn]=ind2sub(size(EJ,1),ind);
                                                % Josephson energy Hamiltonian
                                                HJ0=HJ0-EJ(Dmatrow,Dmatcolumn)/2.*(biasmatrix0(Dmatrow,Dmatcolumn).*Dmat{Dmatrow,Dmatcolumn}+...
                                                    conj(biasmatrix0(Dmatrow,Dmatcolumn)).*Dmat{Dmatrow,Dmatcolumn}');
    %                                                 end
    %                                             end
    %                                         end
                                            end
                                            HJ0=sparse(HJ0)./obj.h*1e-9;
                                            H0=HLC+HJ0;
                                            [ev,en]=eigs(H0,20,'sr');
                                            [en,order]=sort(real(diag(en)));
                                            minus=ev(:,order(2));
                                            plus=ev(:,order(1));
%                                             up0=(plus-minus)./sqrt(2);
%                                             down0=(plus+minus)./sqrt(2);
%                                             if up0'*I*up0>0 && down0'*I*down0<0
%                                                 up=up0;
%                                                 down=down0;
%                                             else
%                                                 up=down0;
%                                                 down=up0;
%                                             end
                                            Ip=abs(minus'*I*plus);
                                            Delta=en(2)-en(1);
                                            trace0=en(2)+en(1);
                                        end
                                        %biasmatrix=obj.getBiasm(sweep,iter); %Update biases from sweep
                                        biasmatrix=obj.getBiasm(fluxbcopy); %Update biases from sweep
                                        for ind=find(EJ)' %Sum over nonzero terms in the EJ matrix
                                            [Dmatrow,Dmatcolumn]=ind2sub(size(EJ,1),ind);
%                                         for Dmatrow=1:nummodes
%                                             for Dmatcolumn=Dmatrow:nummodes
%                                                 if EJ(Dmatrow,Dmatcolumn)~=0 %sum only on the branches containing a JJ
                                            % Josephson energy Hamiltonian
                                            HJ=HJ-EJ(Dmatrow,Dmatcolumn)/2.*(biasmatrix(Dmatrow,Dmatcolumn).*Dmat{Dmatrow,Dmatcolumn}+...
                                                conj(biasmatrix(Dmatrow,Dmatcolumn)).*Dmat{Dmatrow,Dmatcolumn}');
%                                                 end
%                                             end
%                                         end
                                        end
                                        HJ=HJ./obj.h*1e-9;
                                        HJ=sparse(HJ);
                                        switch method
                                            case 1
                                                [ev,en]=eigs(HLC+HJ,20,'sr');
                                                [en,order]=sort(real(diag(en)));
                                                ev=sparse(ev(:,[order(1) order(2)]));
                                                en=diag(en(1:2));
                                                Ip=ncon({conj(ev),I,ev},{[1 -1],[1 2],[2,-2]});
                                                [U,~]=eig(full(Ip));
                                                Hp=U'*en*U;
                                                if any(strcmp({sweep.parameter},'fz'))
                                                    fz=sweep(strcmp({sweep.parameter},'fz')).range(iter);
                                                else
                                                    fz=obj.fluxb.fz;
                                                end
                                                for opind=1:length(opnames)
                                                    if opnames{opind}=='z' && fz-floor(fz)<0.5
                                                        out(iter,opind)=-abs(real(trace(Hp*ops{3})))/2;
                                                    elseif opnames{opind}=='z' && fz-floor(fz)>0.5
                                                        out(iter,opind)=-abs(real(trace(Hp*ops{3})))/2;
                                                    elseif opnames{opind}=='x'
                                                        out(iter,opind)=-sqrt(real(trace(Hp*ops{1}))^2+real(trace(Hp*ops{2}))^2)/2;
                                                    elseif opnames{opind}=='y'
                                                        out(iter,opind)=0;
                                                    elseif opnames{opind}=='I'
                                                        out(iter,opind)=real(trace(Hp*ops{4}))/2;
                                                    end
                                                end
                                            case 2
                                                for opind=1:length(opnames)
                                                    out(iter,opind)=real(trace((HLC+HJ)*ops{opind}))/2;
                                                end
                                            case 3
                                                [ev,en]=eigs(HLC+HJ,20,'sr');
                                                [~,order]=sort(real(diag(en)));
                                                ev=ev(:,[order(1) order(2)]);
                                                ops=obj.getPaulis(opnames,3,I,ev);
                                                H=HLC+HJ;
                                                if addcorr==true
                                                    if iter==1
                                                        Uold=obj.getPaulis('I',3,I,ev);
                                                        Uold=Uold{1};
                                                    else
                                                        U=obj.getPaulis('I',3,I,ev);
                                                        U=U{1};
                                                        dU=length(sweep(1).range).*(U-Uold)./(sweeptime);
                                                        H=H-1i.*dU*U';
                                                        Uold=U;
                                                    end
                                                end
                                                for opind=1:length(opnames)
                                                    out(iter,opind)=real(trace((H)*ops{opind}))/2;
                                                end
                                            case 4
                                                for opind=1:length(opnames)
                                                    if opnames{opind}=='z'
                                                    	out(iter,opind)=-(fzcopy-0.5)*Ip*obj.Phi0/obj.h*1e-9;
                                                    elseif opnames{opind}=='y'
                                                        out(iter,opind)=0;
                                                    elseif opnames{opind}=='x'
                                                        out(iter,opind)=-Delta/2;
                                                    elseif opnames{opind}=='I'
                                                        out(iter,opind)=trace0/2;
                                                    end
                                                end
                                        end
                                    end
                                case 'voltage'
                                    %voltage sweep, only the terms relative to the interaction with the
                                    %bias charge need to be recalculated;
                                    out=zeros(length(sweep(1).range),length(opnames));
                                    HLC=sparse(obj.getHLC(true)); %Linear part of the Hamiltonian
                                    HJ=sparse(obj.getHJ); %Josephson part of the Hamiltonian
                                    Hlen=length(HJ);
                                    R=obj.rotation;
                                    Rlen=length(R);
                                    Cmat=obj.parammat.Cmat;
                                    Cinv=(R/Cmat)*R';
                                    [~,Q]=obj.getPQ();
                                    f=waitbar(1/length(sweep(1).range),strcat('Sweep progress:',num2str(100/length(sweep(1).range),2),'%'));
                                    for iter=1:length(sweep(1).range)
                                        waitbar(iter/length(sweep(1).range),f,strcat('Sweep progress:',num2str(100*iter/length(sweep(1).range),2),'%'));
                                        Hbias=zeros(Hlen);
                                        Qoff=zeros(1,Rlen);
                                        for ind=1:length(sweep)
                                            Qoff(sweep(ind).parameter)=sweep(ind).range(iter)*2*obj.e;
                                            for ind2=1:length(obj.voltb)
                                                if obj.voltb(ind2).node==sweep(ind).parameter
                                                    Qoff(sweep(ind).parameter)=Qoff(sweep(ind).parameter)+obj.voltb(ind2).Qg*2*obj.e;
                                                end
                                            end
                                            if sweep(ind).parameter==2 && isequal(obj.circuitname,"JPSQRF")
                                                q=sweep(ind).range(iter)*2*obj.e;
                                            end
                                        end
                                        Qoff0=Qoff(1)*eye(length(Hlen));
                                        Qoff=inv(R)'*Qoff';
                                        Qoff=Qoff';
                                        if any(Qoff)
                                            for ind=find(Qoff)
                                                Hbias=Hbias+Qoff(ind).*ncon({Cinv(ind,:)',Q},{1,[-1 -2 1]})+...
                                                    Cinv(ind,ind)*(Qoff(ind))^2/2.*eye(Hlen);
                                            end
                                        end
                                        Hbias=sparse(Hbias./obj.h*1e-9);
                                        switch method
                                            case 1
                                                [ev,en]=eigs(HLC+HJ+Hbias,20,'sr');
                                                [en,order]=sort(real(diag(en)));
                                                ev=sparse(ev(:,[order(1) order(2)]));
                                                en=diag(en(1:2));
                                                Ip=ncon({conj(ev),I+Qoff0,ev},{[1 -1],[1 2],[2,-2]});
                                                [U,~]=eig(full(Ip));
                                                Hp=U'*en*U;
                                                if any(strcmp({sweep.parameter},'fz'))
                                                    fz=sweep(strcmp({sweep.parameter},'fz')).range(iter);
                                                else
                                                    fz=obj.fluxb.fz;
                                                end
                                                for opind=1:length(opnames)
                                                    if opnames{opind}=='z' && fz-floor(fz)<0.5
                                                        out(iter,opind)=-abs(real(trace(Hp*ops{3})))/2;
                                                    elseif opnames{opind}=='z' && fz-floor(fz)>0.5
                                                        out(iter,opind)=abs(real(trace(Hp*ops{3})))/2;
                                                    elseif opnames{opind}=='x'
                                                        out(iter,opind)=-sqrt(real(trace(Hp*ops{1}))^2+real(trace(Hp*ops{2}))^2)/2;
                                                    elseif opnames{opind}=='y'
                                                        out(iter,opind)=0;
                                                    elseif opnames{opind}=='I'
                                                        out(iter,opind)=real(trace(Hp*ops{4}))/2;
                                                    end
                                                end
                                            case 2
                                                for opind=1:length(opnames)
                                                    out(iter,opind)=real(trace((HLC+HJ+Hbias)*ops{opind}))/2;
                                                end
                                            case 3
                                                [ev,en]=eigs(HLC+HJ+Hbias,20,'sr');
                                                [~,order]=sort(real(diag(en)));
                                                ev=ev(:,[order(1) order(2)]);
                                                if isequal(obj.circuitname,"JPSQRF")
                                                    ops=obj.getPaulis(opnames,3,I+Qoff0,ev,{},q);
                                                else
                                                    ops=obj.getPaulis(opnames,3,I+Qoff0,ev);
                                                end
                                                for opind=1:length(opnames)
                                                    out(iter,opind)=real(trace((HLC+HJ+Hbias)*ops{opind}))/2;
                                                    %                                                     out(iter,opind)=real(trace(Hpr*ops{opind}))/2;
                                                end
                                        end
                                    end
                            end
                            close(f)
                            if nargin>4 && isequal(plotopt,'Plot') && optionp %Plot results
                                figure
                                plot(sweep(1).range,out,'LineWidth',2);
                                set(gca,'fontsize',18);
                                xlim([min(sweep(1).range),max(sweep(1).range)])
                                if isequal(sweep(1).sweeptype,'voltage')
                                    labelx=strcat('$Q_{g',num2str(sweep(1).parameter),'}$');
                                else
                                    labelx=strcat('\textit{',sweep(1).parameter,'}');
                                end
                                xlabel(labelx,'FontSize',22,'Interpreter','latex')
                                if isequal(sweep(1).sweeptype,'voltage')
                                    labely=strcat('Coefficients $(Q_{g',num2str(sweep(1).parameter),'})$ (GHz)');
                                else
                                    labely=strcat('Coefficients $(',sweep(1).parameter,')$ (GHz)');
                                end
                                ylabel(labely,'FontSize',22,'Interpreter','latex')
                                legend(cellfun(@(x) strcat('$\sigma_',x,'$'),opnames,'UniformOutput',false),'Interpreter','latex')
                                legend('boxoff')
                            end
                        else
                            fprintf("Inconsistent params ranges\n");
                        end
                    end
                end
            end
        end
        
        %% Draws a graph of the circuit
        function circuitgraph(obj)
            switch(obj.circuitname)
                case "fluxRF"
                    figure
                    G=graph([[0 1];[1 0]],{'n0','n1'});
                    p1=plot(G,'EdgeLabel',{'Lq//Ic//Csh'},'LineWidth',2,'NodeFontSize',16,'EdgeFontSize',16);
                    highlight(p1,[1,2],'EdgeColor','r')
                    set(gca,'visible','off')
                case "flux3JJ"
                    figure
                    G=graph([[0,1,1];[1,0,1];[1,1,0]],{'n0','n1','n2'});
                    p1=plot(G,'EdgeLabel',{'Ic','Ic','Ica//Csh'},'LineWidth',2,'NodeFontSize',16,'EdgeFontSize',16);
                    highlight(p1,[2,3],'EdgeColor','r')
                    set(gca,'visible','off')
                case "fluxRF2loops"
                    figure
                    G=graph([[0,1,1,1];[1,0,1,1];[1,1,0,0];[1,1,0,0]],{'n0','n1','n2','n3'});
                    p1=plot(G,'EdgeLabel',{'Lc//Csh','Lco1','Lco2','Ic2','Ic1'},'LineWidth',2,'NodeFontSize',16,'EdgeFontSize',16);
                    highlight(p1,[2,3],'EdgeColor','r')
                    highlight(p1,[2,4],'EdgeColor','r')
                    set(gca,'visible','off')
                case "flux3JJL"
                    figure
                    G=graph([[0,1,1,0];[1,0,0,1];[1,0,0,1];[0,1,1,0]],{'n0','n1','n2','n3'});
                    p1=plot(G,'EdgeLabel',{'Lc','Ic','Ic','Ica//Csh'},'LineWidth',2,'NodeFontSize',16,'EdgeFontSize',16);
                    highlight(p1,[3,4],'EdgeColor','r')
                    set(gca,'visible','off')
                case "flux4JJL"
                    figure
                    G=graph([[0,1,1,0,0,0];[1,0,0,1,1,1];[1,0,0,1,0,0];[0,1,1,0,1,1];[0,1,0,1,0,0];[0,1,0,1,0,0]],{'n0','n1','n2','n3','n4','n5'});
                    p1=plot(G,'EdgeLabel',{'Ic','Lq','Csh','Ica/2','Ica/2','Ic','Lco1','Lco2'},'LineWidth',2,'Layout','layered','Sources',[1 2 6] ,'Sinks',[3 4],'NodeFontSize',16,'EdgeFontSize',16);
                    highlight(p1,[2,5],'EdgeColor','r')
                    highlight(p1,[2,6],'EdgeColor','r')
                    set(gca,'visible','off')
                case "JPSQRF"
                    figure
                    G=graph([[0,1,1,1];[1,0,1,1];[1,1,0,1];[1,1,1,0]],{'n0','n1','n2','n3'});
                    p1=plot(G,'EdgeLabel',{'Cl||Ll','Cg','Cr||Lr','Icl','Csh','Icr'},'LineWidth',2,'NodeFontSize',16,'EdgeFontSize',16);
                    highlight(p1,[2,3],'EdgeColor','r')
                    set(gca,'visible','off')
                case "LCres"
                    figure
                    G=graph([[0 1];[1 0]],{'n0','n1'});
                    plot(G,'EdgeLabel',{'L//C'},'LineWidth',2,'NodeFontSize',16,'EdgeFontSize',16);
                    set(gca,'visible','off')
                otherwise
                    fprintf("Circuit graph not defined for this circuit\n");
            end
        end
        
        %% Calculates plasma frequencies for all modes
        function out=getPlasmaf(obj)
            R=obj.rotation;
            Cmat=obj.parammat.Cmat;
            Cinv=(R/Cmat)*R';
            Lmat=(R'\obj.parammat.Lmat)/R;
            Lmat(abs(Lmat)<1e-3)=0;
            Cdiag=diag(Cinv);
            Ldiag=diag(Lmat);
            fpsquared=Cdiag.*Ldiag;
            out=arrayfun(@sqrt,fpsquared)';
        end
        
        %% Generates bias matrix, where M(i,j)=exp(i*phi^ext_ij) if (i,j) 
        % is a closure branch, M(i,j)=1 otherwise
        function out=getBiasm(obj,fluxbiases,elem) % the optional input is used in flux sweeps 
            % to pass the values of the fluxes at each sweep point.
            % fluxbiases should be a structure of the same form of
            % obj.fluxb
            if isempty(obj.closures)
                out=triu(ones(length(obj.modes)));
            else
                fluxclos={obj.closures.positions};
                fluxclosnames={obj.closures.names};
                out=triu(ones(length(obj.modes)));
                if nargin<2 % if no input is specified
                    for closure=1:length(fluxclos)
                        positions=fluxclos{1,closure};
                        positionx=positions(1);
                        positiony=positions(2);
                        if ischar(fluxclosnames{1,closure}) % if the closure flux is one of the default flux biases (like 'fz')
                        	out(positionx,positiony)=exp(1i*2*pi*obj.fluxb.(fluxclosnames{1,closure}));
                        elseif iscell(fluxclosnames{1,closure}) % if the closure flux is a combination of fluxes (like 'fz'+'fx'/2)
                            fluxvec=zeros(1,length(fluxclosnames{1,closure})-1);
                            for indf=1:length(fluxclosnames{1,closure})-1
                                fluxvec(indf)=obj.fluxb.(fluxclosnames{1,closure}{indf});
                            end
                            fun=fluxclosnames{1,closure}{end};
                            out(positionx,positiony)=exp(1i*2*pi*fun(fluxvec));
                        end                        
                    end
                elseif nargin==2
                    for closure=1:length(fluxclos)
                        positions=fluxclos{1,closure};
                        positionx=positions(1);
                        positiony=positions(2);
                        if ischar(fluxclosnames{1,closure}) % if the closure flux is one of the default flux biases (like 'fz')
                        	out(positionx,positiony)=exp(1i*2*pi*fluxbiases.(fluxclosnames{1,closure}));
                        elseif iscell(fluxclosnames{1,closure}) % if the closure flux is a combination of fluxes (like 'fz'+'fx'/2)
                            fluxvec=zeros(1,length(fluxclosnames{1,closure})-1);
                            for indf=1:length(fluxclosnames{1,closure})-1
                                fluxvec(indf)=fluxbiases.(fluxclosnames{1,closure}{indf});
                            end
                            fun=fluxclosnames{1,closure}{end};
                            out(positionx,positiony)=exp(1i*2*pi*fun(fluxvec));
                        end
                    end
                else %use legacy version for qubitsinteract
                    sweepstruct=fluxbiases;
                    if isequal(obj.circuitname,"fluxRF2loops") ||  isequal(obj.circuitname,"squidres")%the two loops (CJJ)
                    % RF squid qubit requires the fz bias to be balanced whenever
                    % fx!=0, this takes care that this is the case
                    if length(sweepstruct)==1
                        switch(sweepstruct.parameter)
                            case 'fz'
                                fz=sweepstruct.range(elem);
                                fx=obj.fluxb.fx;
                            case 'fx'
                                fx=sweepstruct.range(elem);
                                fz=obj.fluxb.fz;
                        end
                    elseif length(sweepstruct)==2
                        fz=0;
                        fx=0;
                        for paramn=1:2
                            switch(sweepstruct(paramn).parameter)
                                case 'fz'
                                    fz=fz+sweepstruct(paramn).range(elem);
                                case 'fx'
                                    fx=fx+sweepstruct(paramn).range(elem);
                            end
                        end
                    end
                    out(1,2)=exp(1i*2*pi*(fz-fx/2));
                    out(1,3)=exp(1i*2*pi*(fz+fx/2));
                elseif isequal(obj.circuitname,"JPSQRF2loops")
                        if length(sweepstruct)==1
                            switch(sweepstruct.parameter)
                                case 'fz'
                                    fz=sweepstruct.range(elem);
                                    fL=obj.fluxb.fL;
                                    fR=obj.fluxb.fR;
                                case 'fL'
                                    fL=sweepstruct.range(elem);
                                    fz=obj.fluxb.fz;
                                    fR=obj.fluxb.fR;
                                case 'fR'
                                    fR=sweepstruct.range(elem);
                                    fz=obj.fluxb.fz;
                                    fL=obj.fluxb.fL;
                            end
                        elseif length(sweepstruct)==2
                            fz=0;
                            fR=0;
                            fL=0;
                            bias3=setdiff({'fz','fR','fL'},{sweepstruct.parameter});
                            switch bias3{1}
                                case 'fz'
                                    fz=obj.fluxb.fz;
                                case 'fL'
                                    fL=obj.fluxb.fL;
                                case 'fR'
                                    fR=obj.fluxb.fR;
                            end
                            for paramn=1:2
                                switch(sweepstruct(paramn).parameter)
                                    case 'fz'
                                        fz=fz+sweepstruct(paramn).range(elem);
                                    case 'fL'
                                        fL=fL+sweepstruct(paramn).range(elem);
                                    case 'fR'
                                        fR=fR+sweepstruct(paramn).range(elem);
                                end
                            end
                        elseif length(sweepstruct)==3
                            fz=0;
                            fL=0;
                            fR=0;
                            for paramn=1:3
                                switch(sweepstruct(paramn).parameter)
                                    case 'fz'
                                        fz=fz+sweepstruct(paramn).range(elem);
                                    case 'fL'
                                        fL=fL+sweepstruct(paramn).range(elem);
                                    case 'fR'
                                        fR=fR+sweepstruct(paramn).range(elem);
                                end
                            end
                        end
                        out(2,5)=exp(1i*2*pi*fR);
                        out(2,4)=exp(1i*2*pi*(-fz-fL/2+fR/2));
                        out(2,3)=exp(1i*2*pi*(-fz+fL/2+fR/2));
                    elseif isequal(obj.circuitname,"flux4JJL")
                        if length(sweepstruct)==1
                            switch(sweepstruct.parameter)
                                case 'fz'
                                    fz=sweepstruct.range(elem);
                                    fx=obj.fluxb.fx;
                                case 'fx'
                                    fx=sweepstruct.range(elem);
                                    fz=obj.fluxb.fz;
                            end
                        elseif length(sweepstruct)==2
                            fz=0;
                            fx=0;
                            for paramn=1:2
                                switch(sweepstruct(paramn).parameter)
                                    case 'fz'
                                        fz=fz+sweepstruct(paramn).range(elem);
                                    case 'fx'
                                        fx=fx+sweepstruct(paramn).range(elem);
                                end
                            end
                        end
                        out(1,4)=exp(1i*2*pi*(fz-fx/2));
                        out(1,5)=exp(1i*2*pi*(fz+fx/2));
                    else
                        for closure=1:length(fluxclos)
                            for param=1:length(sweepstruct)
                                if isequal(sweepstruct(param).parameter,fluxclosnames{1,closure})
                                    positions=[fluxclos{1,closure}];
                                    positionx=positions(1);
                                    positiony=positions(2);
                                    out(positionx,positiony)=exp(1i*2*pi*sweepstruct(param).range(elem));
                                else
                                    positions=[fluxclos{1,closure}];
                                    positionx=positions(1);
                                    positiony=positions(2);
                                    out(positionx,positiony)=exp(1i*2*pi*obj.fluxb.(fluxclosnames{1,closure}));
                                end
                            end
                        end
                    end
                end
            end
        end
        
        %% Displays eigenfunctions
        function [out,phi]=wavefunc(obj,state)
            if isequal(obj.circuitname,"fluxRF")
                fz=obj.fluxb.fz*2*pi;
                f=@(n,x) hermiteH(n,x)./sqrt(2^n*factorial(n));
                H=obj.getH;
                [P,~]=obj.getPQ;
                p0=sqrt(2)/P(1,2)*obj.Phi0/(2*pi);
                [evect,evals]=eigs(H,state,'sr');
                [~,order]=sort(real(diag(evals)));
                eigenstate=evect(:,order(state));
                eigenstate=eigenstate./eigenstate(find(abs(eigenstate)>1e-7,1))*abs(eigenstate(find(abs(eigenstate)>1e-7,1)));
                phi=real(eigenstate'*P*eigenstate)/obj.Phi0+obj.fluxb.fz;
                if norm(imag(eigenstate))<1e-4
                    eigenstate=real(eigenstate);
                    eigenstate=eigenstate./norm(eigenstate);
                end
                x=linspace(-pi,pi,100);
                out=zeros(1,length(x));
                for n=1:length(eigenstate)
                    out=out+eigenstate(n).*f(n,p0*x);
                end
                out=pi^(-0.25)*exp(-(p0*x).^2/2).*out;
                % out=abs(out).^2;
                out=out./norm(out);
                figure
                plot(linspace(fz-pi,fz+pi,length(x))./(2*pi),out,'Linewidth',2)
                hold
                plot(linspace(fz-pi,fz+pi,length(x))./(2*pi),abs(out).^2,'Linewidth',2)
                xlim(round([(fz-pi)/(2*pi) (fz+pi)/(2*pi)],1))
                set(gca,'fontsize',18);
                xlabel('$\phi/2\pi$','FontSize',22,'Interpreter','latex')
                ylab=strcat('$\Psi_',num2str(state),'(\phi)$');
                leglab=strcat('$|\Psi_',num2str(state),'(\phi)|^2$');
                ylabel(ylab,'FontSize',22,'Interpreter','latex')
                legend({ylab,leglab},'Location','southeast','FontSize',15,'Interpreter','latex')
                legend('boxoff');
            else
                fprintf("Function not available for this circuit\n");
            end
        end
        
        %% Calculates P and Q operators
        % These are the flux and charge operators associated with each mode
        % (there are as many modes as there are nodes in the circuit graph)
        % the operators are returned as a list of matrices (i.e. a 3d
        % array) as long as the number of modes, or, as a cell array, if
        % smalloutswitch is true
        function [P,Q]=getPQ(obj,smalloutswitch,shiftswitch) %flux and charge operators are given as 3-dimentsional arrays
            truncation=obj.truncations;
            R=obj.rotation;
            Cmat=obj.parammat.Cmat;
            Cinv=(R/Cmat)*R';
            Lmat=(R'\obj.parammat.Lmat)/R;
            Qoff=zeros(1,length(R));
            if ~isempty(obj.voltb) && nargin==3 && shiftswitch==true
                for ind=1:length(obj.voltb)
                    Qoff(obj.voltb(ind).node)=Qoff(obj.voltb(ind).node)+obj.voltb(ind).Qg*2*obj.e;
                end
                Qoff=inv(R)'*Qoff';
                Qoff=Qoff';
            end
%             Lmat(abs(Lmat)<1e-3)=0;
            for ntrunc=1:length(truncation)
                if (obj.modes(ntrunc)=="Jos") || (obj.modes(ntrunc)=="JosPhase") || (obj.modes(ntrunc)=="Island")
                    truncation(ntrunc)=2*truncation(ntrunc)+1; %2N+1 states for Josephson and Island modes,
                    % makes sure that the identity matrices for every mode
                    % have the right diemensions
                end
            end
            nummodes=length(obj.modes);
            Id=cellfun(@eye,num2cell(truncation),'UniformOutput',0); %cell of identity matrices with the appropriate dimension for each mode.
            %Used to calculate kronecker products
            if nargin<2 || (nargin>=2 && smalloutswitch==false)
                P=zeros(prod(truncation),prod(truncation),nummodes); % Flux and charge operators for every mode
                Q=zeros(prod(truncation),prod(truncation),nummodes);
            elseif nargin>=2 && smalloutswitch==true
                P=Id;
                Q=Id;
            end
            for modenum=1:nummodes
                if obj.modes(modenum)=="HO"
                    % for harmonic oscillator modes, express P and Q in
                    % terms of ladder operators. The matrices are truncated
                    % at a given occupation number
                    Psmall=Id;
                    Qsmall=Id;
                    diagonal=sqrt(1:truncation(modenum));
                    diagM=diag(diagonal);
                    a=circshift(diagM,1,2);
                    a(:,1)=0;
                    adagger=a';
                    Z=sqrt(Cinv(modenum,modenum)/Lmat(modenum,modenum));
                    Psmall(1,modenum)={sqrt(obj.h*Z/(4*pi))*(a+adagger)};
                    Qsmall(1,modenum)={1i*sqrt(obj.h/(4*pi*Z))*(adagger-a)+Qoff(1,modenum)*eye(truncation(modenum))};
                    if nargin<2 || (nargin>=2 && smalloutswitch==false)
                        % expressions of the P and Q operators in the
                        % composite Hilbert space obtained by taking the
                        % outer product of each mode Hilbert space
                        P(:,:,modenum)=superkron(Psmall);
                        Q(:,:,modenum)=superkron(Qsmall);
                    elseif nargin>=2 && smalloutswitch==true
                        % expressions of the P and Q operators in the
                        % single mode Hilbert spaces
                        P(1,modenum)=Psmall(1,modenum);
                        Q(1,modenum)=Qsmall(1,modenum);
                    end
                elseif (obj.modes(modenum)=="Jos") || (obj.modes(modenum)=="Island")
                    % For Josephson and island modes, only the charge
                    % operators are necessary. These are expressed in the
                    % charge basis, hence they are diagonal. Matrices are
                    % truncated at a given absolute charge (in integer
                    % multiples of 2e: Q=diag(-2Ne,...,2Ne))
                    Qsmall=Id;
                    %diagM=diag(fliplr(-obj.truncations(modenum):obj.truncations(modenum)));
                    diagM=diag(-obj.truncations(modenum):obj.truncations(modenum));
                    Qsmall(1,modenum)={-2*obj.e*diagM+Qoff(1,modenum)*eye(truncation(modenum))};
%                     Q(:,:,modenum)=superkron(Qsmall); %Flux operators do not appear for Josephson and island modes
                    if nargin<2 || (nargin>=2 && smalloutswitch==false)
                        Q(:,:,modenum)=superkron(Qsmall);
                    elseif nargin>=2 && smalloutswitch==true
                        Q(1,modenum)=Qsmall(1,modenum);
                    end
                elseif (obj.modes(modenum)=="JosPhase")
                    % Alternatively use Phase basis for Josephson modes:
                    Nmax=obj.truncations(modenum);
                    n0=linspace(-Nmax,Nmax,2*Nmax+1);
                    theta=2*pi*n0/(2*Nmax+1);
                    n=zeros(2*Nmax+1); %Charge states
                    for k=1:length(n)
                        n(k,:)=sqrt(1/(2*Nmax+1))*exp(-1i*n0(k).*theta);
                    end
                    n=n.';
                    N=ncon({n,diag(n0),conj(n)},{[-1 1],[1 2],[-2,2]});
                    N=(N+N')./2;
                    N=1i.*imag(N);
                    Qsmall=Id;
                    Qsmall(1,modenum)={-2*obj.e*N+Qoff(1,modenum)*eye(truncation(modenum))};
%                     Q(:,:,modenum)=superkron(Qsmall); %Flux operators do not appear for Josephson and island modes
                    if nargin<2 || (nargin>=2 && smalloutswitch==false)
                        Q(:,:,modenum)=superkron(Qsmall);
                    elseif nargin>=2 && smalloutswitch==true
                        Q(1,modenum)=Qsmall(1,modenum);
                    end
                end
            end
        end
        
        %% Calculates N operators, used to write the diagonal terms of the
        % linear part of the Hamiltonian (h.o. modes only) = hbar*omega*N
        function N=getNops(obj)
            truncation=obj.truncations;
            for ntrunc=1:length(truncation)
                if (obj.modes(ntrunc)=="Jos") || (obj.modes(ntrunc)=="JosPhase") || (obj.modes(ntrunc)=="Island")
                    truncation(ntrunc)=2*truncation(ntrunc)+1; %2N+1 states for Josephson and Island modes
                    % makes sure that the identity matrices for every mode
                    % have the right diemensions
                end
            end
            nummodes=length(obj.modes);
            N=zeros(prod(truncation),prod(truncation),nummodes);
            Id=arrayfun(@eye,truncation,'UniformOutput',0);
            for modenum=1:nummodes
                if obj.modes(modenum)=="HO" %Number operators only appear for oscillator modes
                    Nsmall=Id;
                    Nsmall(1,modenum)={diag(0:(truncation(modenum)-1))};
                    N(:,:,modenum)=superkron(Nsmall);
                end
            end
        end
        
        %% Calculates Displacement operators for every branch with a JJ
        % these represent the term exp(i*phi_ab) of the cosine appearing in
        % the Josephson energy relative to the junction on the branch ab.
        % phi_ab=2*pi/Phi_0*(Phi_a-Phi_b), where Phi_i is the node flux
        % associated with the node i.
        function Dmat=getDops(obj)
            truncation=obj.truncations;
            for ntrunc=1:length(truncation)
                if (obj.modes(ntrunc)=="Jos") || (obj.modes(ntrunc)=="JosPhase") || (obj.modes(ntrunc)=="Island")
                    truncation(ntrunc)=2*truncation(ntrunc)+1; %2N+1 states for Josephson and Island modes
                    % makes sure that the identity matrices for every mode
                    % have the right diemensions
                end
            end
            R=obj.rotation;
            Rinv=inv(R);
            Cmat=obj.parammat.Cmat;
            Cinv=(R/Cmat)*R';
            Lmat=(R'\obj.parammat.Lmat)/R;
%             Lmat(abs(Lmat)<1e-3)=0;
            EJ=obj.parammat.Ejmat; % matrix containing the josephson energies of the junctions on each branch
                                   % connecting two given nodes
            nummodes=length(obj.modes);
            D=cell(1,nummodes);
            Dmat=cell(nummodes,nummodes);
            P=cell(1,nummodes);
            for modenum=1:nummodes
                if obj.modes(modenum)=="HO" %for an HO mode k, D_k=expm(1i*P_k)
                    diagonal=sqrt(1:truncation(modenum));
                    diagM=diag(diagonal);
                    a=circshift(diagM,1,2);
                    a(:,1)=0;
                    adagger=a';
                    Z=sqrt(Cinv(modenum,modenum)/Lmat(modenum,modenum));
                    P(1,modenum)={2*pi/obj.Phi0*sqrt(obj.h*Z/(4*pi)).*(a+adagger)};
                elseif obj.modes(modenum)=="Jos"
                    Dp=circshift(eye(truncation(modenum)),-1,2);
                    Dp(:,end)=0;
                    D(1,modenum)={Dp};
                elseif obj.modes(modenum)=="Island"
                    D(1,modenum)={eye(truncation(modenum))};
                end
            end
%             for Dmatrow=1:nummodes %D(i,j)==exp(1i.*(P_i-P_j))=exp(1i.*sum_k((R)^-1_ik-(R)^-1_jk)P'_k),
                % and D(i,i)==exp(1i.*P_i)=exp(1i.*sum_k((R)^-1_ik*P'_k))=prod_k(exp(1i.*(R)^-1_ik*P'_k)),
                % where
                % the P_k's are the original modes (corresponding to the
                % circuit graph) and the P'_k's are the ones after the
                % rotation R
            for ind=find(EJ)'
                [Dmatrow,Dmatcolumn]=ind2sub(size(EJ,1),ind);
                %                 for Dmatcolumn=Dmatrow:nummodes
%                 if EJ(Dmatrow,Dmatcolumn)~=0
                Dcopy=cell(1,nummodes);
                for modenum=1:nummodes
                    if Dmatcolumn==Dmatrow
                        exponent=Rinv(Dmatrow,modenum);
                    else
                        exponent=Rinv(Dmatcolumn,modenum)-Rinv(Dmatrow,modenum);
                    end
                    if obj.modes(modenum)=="HO"
                        Dcopy(1,modenum)={expm(1i*exponent.*P{1,modenum})};
                        %Dcopy(1,modenum)={expm((exponent)^2.*P{1,modenum}{1}/2)*expm(1i*exponent.*P{1,modenum}{2})*expm(1i*exponent.*P{1,modenum}{3})};
                    elseif obj.modes(modenum)=="Jos"
                        if exponent==-1 %exponent-1<1e-5
                            Dcopy(1,modenum)=D(1,modenum);
                        elseif exponent==1 %exponent+1<1e-5
                            Dcopy(1,modenum)={D{1,modenum}'};
                        elseif exponent==0 %abs(exponent)<1e-7
                            Dcopy(1,modenum)={eye(truncation(modenum))};
                            %                                 else
                            %                                     fprintf('Error\n')
                        end
                    elseif obj.modes(modenum)=="JosPhase"
                        Nmax=obj.truncations(modenum);
                        n0=linspace(-Nmax,Nmax,2*Nmax+1);
                        theta=2*pi*n0/(2*Nmax+1);
                        if exponent==1 %exponent-1<1e-5
                            Dcopy(1,modenum)={diag(exp(1i.*theta))};
                        elseif exponent==-1 %exponent+1<1e-5
                            Dcopy(1,modenum)={diag(exp(-1i.*theta))};
                        elseif exponent==0 %abs(exponent)<1e-7
                            Dcopy(1,modenum)={eye(truncation(modenum))};
                            %                                 else
                            %                                     fprintf('Error\n')
                        end
                    elseif obj.modes(modenum)=="Island"
                        Dcopy(1,modenum)=D(1,modenum);
                    end
                end
                Dmat(Dmatrow,Dmatcolumn)={superkron(Dcopy)};
%                 end
%                 end
            end
        end
        
        %% Computes capacitive and inductive energy Ham
        function out=getHLC(obj,biastoggle)
            if nargin<2
                biastoggle=false; %leave out voltage bias terms for voltage sweeps
            end
            truncation=obj.truncations;
            for ntrunc=1:length(truncation)
                if (obj.modes(ntrunc)=="Jos") || (obj.modes(ntrunc)=="JosPhase") || (obj.modes(ntrunc)=="Island")
                    truncation(ntrunc)=2*truncation(ntrunc)+1; %2N+1 states for Josephson and Island modes
                    % makes sure that the identity matrices for every mode
                    % have the right diemensions
                end
            end
            Id=cellfun(@eye,num2cell(truncation),'UniformOutput',0); %cell of identity matrices of edge truncation(modenum)
            [P,Q]=obj.getPQ(true);
            N=obj.getNops;
            R=obj.rotation;
            omegap=obj.getPlasmaf;
            Cmat=obj.parammat.Cmat;
            Cinv=(R/Cmat)*R';
            Lmat=(R'\obj.parammat.Lmat)/R;
%             Lmat(abs(Lmat)<1e-3)=0;
            for modenum=1:length(Cinv)
                if obj.modes(modenum)=="HO"
                    Cinv(modenum,modenum)=0;
                    Lmat(modenum,modenum)=0;%Eliminates the diagonal terms for the HO modes,
                    %these will be replaced by terms of the form
                    %hbar*omegap_k*N_k, where N_k is the number operator
                    %for mode k
                    % Lmat(modenum,modenum)=0;
                end
            end
%             out=zeros(size(N,1));
            out=ncon({omegap',N},{1,[-1 -2 1]}); %diagonal terms sum_k(hbar*omegap_k*N_k)
            out=obj.h/(2*pi).*(out+sum(omegap)/2.*eye(size(N,1))); %convert to Joules and add constant offset hbar*omega/2
%             out=out+0.5.*ncon({Cinv,Q,Q},{[1 2],[-1 3 1],[3 -2 2]},[1 2 3]); %off diagonal terms
%             out=out+0.5.*ncon({Lmat,P,P},{[1 2],[-1 3 1],[3 -2 2]},[1 2 3]);
            for row=1:length(Cinv)
                for col=row:length(Cinv)
                    if row==col %additional diagonal terms
                        if Cinv(row,row)~=0
                            Qcopy=Id;
                            Qcopy(1,row)={Q{1,row}^2};
                            % out=out+0.5*Cinv(row,col).*Q(:,:,row)*Q(:,:,col); 
                            out=out+0.5*Cinv(row,row).*superkron(Qcopy);
                        end
                        if Lmat(row,row)~=0
                            Pcopy=Id;
                            Pcopy(1,row)={P{1,row}^2};
                            % out=out+0.5*Lmat(row,col).*P(:,:,row)*P(:,:,col);
                            out=out+0.5*Lmat(row,row).*superkron(Pcopy);
                        end
                    else %off-diagonal terms
                        if Cinv(row,col)~=0
                            Qcopy=Id;
                            Qcopy(1,col)=Q(1,col);
                            Qcopy(1,row)=Q(1,row);
                            % out=out+0.5*Cinv(row,col).*Q(:,:,row)*Q(:,:,col);
                            out=out+Cinv(row,col).*superkron(Qcopy); 
                        end
                        if Lmat(row,col)~=0
                            Pcopy=Id;
                            Pcopy(1,col)=P(1,col);
                            Pcopy(1,row)=P(1,row);
                            % out=out+0.5*Lmat(row,col).*P(:,:,row)*P(:,:,col);
                            out=out+Lmat(row,col).*superkron(Pcopy);
                        end
                    end
                end
            end
            if ~isempty(obj.voltb) && ~biastoggle %Add the voltage bias term
                Cmat=obj.parammat.Cmat;
                Cinv=(R/Cmat)*R';
                Q1=zeros(length(out),length(out),length(R));
                for mode=1:length(R)
                    Qcopy=Id;
                    Qcopy(1,mode)=Q(1,mode);
                    Q1(:,:,mode)=superkron(Qcopy);
                end
                Qoff=zeros(1,length(R));
                for ind=1:length(obj.voltb)
                    Qoff(obj.voltb(ind).node)=Qoff(obj.voltb(ind).node)+obj.voltb(ind).Qg*2*obj.e;
                end
                Qoff=inv(R)'*Qoff';
                Qoff=Qoff';
                Hextra=zeros(length(out));
                if any(Qoff)
                    for ind=find(Qoff)
                        Hextra=Hextra+Qoff(ind).*ncon({Cinv(ind,:)',Q1},{1,[-1 -2 1]})+...
                            Cinv(ind,ind)*(Qoff(ind))^2/2.*eye(length(out));
                    end
                end
                out=out+Hextra;
            end
            out=out./obj.h*1e-9; %convert to GHz units
        end
        
        %% Computes Josephson energy Ham
        function out=getHJ(obj)
%             nummodes=length(obj.modes);
            EJ=obj.parammat.Ejmat;
            Dmat=obj.getDops;
            biasmatrix=obj.getBiasm;
            truncation=obj.truncations;
            for ntrunc=1:length(truncation)
                if (obj.modes(ntrunc)=="Jos") || (obj.modes(ntrunc)=="JosPhase") || (obj.modes(ntrunc)=="Island")
                    truncation(ntrunc)=2*truncation(ntrunc)+1; %2N+1 states for Josephson and Island modes
                    % makes sure that the identity matrices for every mode
                    % have the right diemensions
                end
            end
            out=zeros(prod(truncation));
            for ind=find(EJ)'
                [Dmatrow,Dmatcolumn]=ind2sub(size(EJ,1),ind);
%             for Dmatrow=1:nummodes
%                 for Dmatcolumn=Dmatrow:nummodes
%                     if EJ(Dmatrow,Dmatcolumn)~=0 %for branches containing a JJ
                out=out-EJ(Dmatrow,Dmatcolumn)/2.*(biasmatrix(Dmatrow,Dmatcolumn).*Dmat{Dmatrow,Dmatcolumn}+...
                    conj(biasmatrix(Dmatrow,Dmatcolumn)).*Dmat{Dmatrow,Dmatcolumn}');
%                     end
%                 end
            end
            out=out+eye(prod(truncation)).*sum(sum(EJ)); % add constant offsets EJ
            out=out./obj.h*1e-9; %convert to GHz units
        end

        %% Calculate Hamiltonian
        function out=getH(obj)
            HLC=obj.getHLC;
            HJ=obj.getHJ;
            out=HLC+HJ;
        end

        %% Calculates branch current operators
        function Ib=getIops(obj,opname)
            opnamecopies=cell(1,length(obj.oplist));
            opnamecopies(1,:)={opname};
            if sum(cellfun(@(x,y) isequal(x,y),obj.oplist,opnamecopies))
                Rinv=inv(obj.rotation);
                if 0%isequal(obj.circuitname,"flux3JJ") && isequal(opname,'Iz')
                    D=obj.getDops;
                    D=D{1,1};
                    Ib=-1i*obj.parameters.Ic/2*(D'-D);
                else
                    ioperator=obj.ioperators.(opname);
                    EJ=obj.parammat.Ejmat;
                    if isequal(class(ioperator{end}),'function_handle')
                        f=ioperator{end};
                        Ib=cell(1,length(ioperator)-1);
                        [P,~]=obj.getPQ;
                        for termn=1:length(ioperator)-1
                            Ibtemp=zeros(length(P(:,:,1)));
                            if ioperator{termn}{1}<ioperator{termn}{2}
                                fprintf("Invalid branch specification: the first index should be greater than the second\n");
                            else
                                if ioperator{termn}{2}==0
                                    ioperator2=[ioperator{termn}{1},ioperator{termn}{1}];
                                else
                                    ioperator2=[ioperator{termn}{1},ioperator{termn}{2}];
                                end
                                if EJ(ioperator2(2),ioperator2(1))==0 %If the branch is inductive
                                    Rvec=Rinv(ioperator{termn}{1},:);
                                    Ibtemp=Ibtemp+ncon({Rvec',P},{1,[-1 -2 1]});
                                    if ioperator{termn}{2}~=0
                                        Rvec=Rinv(ioperator{termn}{2},:);
                                        Ibtemp=Ibtemp-ncon({Rvec',P},{1,[-1 -2 1]});
                                        Ib(1,termn)={abs(obj.parammat.Lmat(ioperator2(1),ioperator2(2))).*Ibtemp};
                                    else
                                        Ib(1,termn)={sum(obj.parammat.Lmat(ioperator2(1),:)).*Ibtemp};
                                    end
                                else
                                    Dmat=obj.getDops;
                                    Ib(1,termn)=1i*pi*EJ(ioperator2(2),ioperator2(1))/obj.Phi0.*(biasmatrix(ioperator2(2),ioperator2(1)).*Dmat{ioperator2(2),ioperator2(1)}-...
                                            conj(biasmatrix(ioperator2(2),ioperator2(1))).*Dmat{ioperator2(2),ioperator2(1)});
                                end
                            end  
                        end
                        Ib=f(Ib{1,:});
                    else
                        [P,~]=obj.getPQ;
                        Ib=zeros(length(P(:,:,1)));
                        if ioperator{1}<ioperator{2}
                            fprintf("Invalid branch specification: the first index should be greater than the second\n");
                        else
                            if ioperator{2}==0
                                ioperator2=[ioperator{1},ioperator{1}];
                            else
                                ioperator2=[ioperator{1},ioperator{2}];
                            end
                            if EJ(ioperator2(2),ioperator2(1))==0 %If the branch is inductive
                                Rvec=Rinv(ioperator{1},:);
                                Ib=Ib+ncon({Rvec',P},{1,[-1 -2 1]});
                                if ioperator{2}~=0
                                    Rvec=Rinv(ioperator{2},:);
                                    Ib=Ib-ncon({Rvec',P},{1,[-1 -2 1]});
                                    Ib=abs(obj.parammat.Lmat(ioperator2(1),ioperator2(2))).*Ib;
                                else
                                    Ib=sum(obj.parammat.Lmat(ioperator2(1),:)).*Ib;
                                end
                            else
                                Dmat=obj.getDops;
                                Ib=1i*pi*EJ(ioperator2(2),ioperator2(1))/obj.Phi0.*(Dmat{ioperator2(2),ioperator2(1)}-Dmat{ioperator2(2),ioperator2(1)}');
                           end
                        end    
                    end
                end
            else
                fprintf("Invalid/undefined current operator\n");
            end
        end
        
        %% Generate Pauli operators from the computational basis states
        function out=getPaulis(obj,opname,method,Iop,states,trunc,q)
            out=cell(1,length(opname));
            if nargin<7 && isequal(obj.circuitname,"JPSQRF")
                q=obj.voltb([obj.voltb.node]==2).Qg;
            end
            switch method
                case 1
                    for opind=1:length(opname)
                        switch opname{opind}
                            case 'I'
                                out{opind}=eye(2);
                            case 'z'
                                out{opind}=[[1,0];[0,-1]];
                            case 'y'
                                out{opind}=[[0,-1i];[1i,0]];
                            case 'x'
                                out{opind}=[[0,1];[1,0]];
                        end
                    end
                case 2
                    if isequal(obj.type,"flux")
                        fzold=obj.fluxb.fz;
                        obj.fluxb.fz=0.5;
                    else
                        Qgold=obj.voltb.Qg;
                        obj.voltb.Qg=0.5;
                    end
                    H=sparse(obj.getH);
                    if isequal(obj.type,"flux")
                        I=sparse(obj.getIops('Iz'));
                    elseif isequal(obj.type,"charge")
                        [~,I]=obj.getPQ(false,true);
                        if length(size(I))==3
                            Rinv=obj.rotation';
                            I=squeeze(ncon({Rinv(1,:),I},{[-1,1],[-2 -3 1]})); %Main qubit island must be on node 1!
                            I=sparse(I); 
                        else
                            I=sparse(I);
                        end
                    end
                    [ev,en]=eigs(H,20,'sr');
                    [~,order]=sort(real(diag(en)));
                    minus=ev(:,order(2));
                    plus=ev(:,order(1));
                    up0=(plus-minus)./sqrt(2);
                    down0=(plus+minus)./sqrt(2);
                    if up0'*I*up0>0 && down0'*I*down0<0
                        up=up0;
                        down=down0;
                    else
                        up=down0;
                        down=up0;
                    end
                    Id=up*up'+down*down';
                    sz=up*up'-down*down';
                    sy=-1i.*up*down'+1i.*down*up';
                    sx=up*down'+down*up';
                    for opind=1:length(opname)
                        switch opname{opind}
                            case 'I'
                                out{opind}=Id;
                            case 'z'
                                out{opind}=sz;
                            case 'y'
                                out{opind}=sy;
                            case 'x'
                                out{opind}=sx;
                        end
                    end
                    if isequal(obj.type,"flux")
                        obj.fluxb.fz=fzold;
                    else
                        obj.voltb.Qg=Qgold;
                    end
                case 3
                    Ip=ncon({conj(states),Iop,states},{[1 -1],[1 2],[2,-2]});
                    [U,Ipn]=eig(Ip);
                    U(:,1)=U(:,1)./U(1,1)*abs(U(1,1));
                    U(:,2)=U(:,2)./U(1,2)*abs(U(1,2));
                    Ipn=diag(Ipn);
                    upcoeff=U(:,Ipn>0);
                    downcoeff=U(:,Ipn<0);
                    if nargin>5 && ~isempty(trunc)
                        E0=zeros(trunc,1);
                        E0(1)=1;
                        E1=zeros(trunc,1);
                        E1(2)=1;
                        up=upcoeff(1)*E0+upcoeff(2)*E1;
                        down=downcoeff(1)*E0+downcoeff(2)*E1;
                    else
                        up=upcoeff(1)*states(:,1)+upcoeff(2)*states(:,2);
                        down=downcoeff(1)*states(:,1)+downcoeff(2)*states(:,2);
                    end
                    if length(opname)==1
                        switch opname
                            case 'I'
                                out={up*up'+down*down'};
                            case 'z'
                                out={up*up'-down*down'};
                            case 'y'
                                if isequal(obj.circuitname,"JPSQRF")
                                   sz=up*up'-down*down';
                                   sy=-1i.*up*down'+1i.*down*up';
                                   out={expm(-1i*pi*(q-obj.e)/(2*obj.e)/2.*sz)*sy*expm(1i*pi*(q-obj.e)/(2*obj.e)/2.*sz)};
                                else
                                    out={-1i.*up*down'+1i.*down*up'};
                                end
                            case 'x'
                                if isequal(obj.circuitname,"JPSQRF")
                                   sz=up*up'-down*down';
                                   sx=up*down'+down*up';
                                   out={expm(-1i*pi*(q-obj.e)/(2*obj.e)/2.*sz)*sx*expm(1i*pi*(q-obj.e)/(2*obj.e)/2.*sz)};
                                else
                                    out={up*down'+down*up'};
                                end
                        end
                    else
                        for opind=1:length(opname)
                            switch opname{opind}
                                case 'I'
                                    out{opind}=up*up'+down*down';
                                case 'z'
                                    out{opind}=up*up'-down*down';
                                case 'y'
                                    if isequal(obj.circuitname,"JPSQRF") && 0
                                       sz=up*up'-down*down';
                                       sy=-1i.*up*down'+1i.*down*up';
                                       theta=pi/2*sin(2*pi*q/(2*obj.e));
                                       out{opind}=expm(1i*theta/2.*sz)*sy*expm(-1i*theta/2.*sz);
                                    else
                                        out{opind}=-1i.*up*down'+1i.*down*up';
                                    end
                                case 'x'
                                    if isequal(obj.circuitname,"JPSQRF") && 0
                                       sz=up*up'-down*down';
                                       sx=up*down'+down*up';
                                       theta=pi/2*sin(2*pi*q/(2*obj.e));
                                       out{opind}=expm(1i*theta/2.*sz)*sx*expm(-1i*theta/2.*sz);
                                    else
                                        out{opind}=up*down'+down*up';
                                    end
                            end
                        end
                    end
            end
        end
        
        function [zero,one]=getComputstates(obj)
            H=obj.getH();
            if isequal(obj.type,"flux")
                Iop=sparse(obj.getIops('Iz'));
            elseif isequal(obj.type,"charge")
                [~,Iop]=obj.getPQ(false,true);
                if length(size(I))==3
                    Rinv=obj.rotation';
                    Iop=squeeze(ncon({Rinv(1,:),Iop},{[-1,1],[-2 -3 1]})); %Main qubit island must be on node 1!
                    Iop=sparse(Iop); 
                else
                    Iop=sparse(Iop);
                end
            end
            [ev,~]=eigs(H,20,'SA');
            sz=obj.getPaulis('z',3,Iop,ev(:,1:2));
            [zero,~]=eigs(sz{1},1,0.9);
            [one,~]=eigs(sz{1},1,-0.9);
        end
        
%     end
%     
%     methods (Access=private)
        
        %% Updates modes according to new rotation matrix R
        % if a new modes rotation is introduced, this function determines
        % the type of every new mode
        function Rprime=updatemodes(obj,R)
            if isequal(R,"Auto") %if the new rotation is defined as "Auto", then
                % this function aumomatically defines a rotation which
                % diagonalises the inverse inductance matrix (so as to
                % isolate the HO modes)
                [Rpinv,Lmat]=eig(obj.parammat.Lmat);
                Lmat(abs(Lmat)<1e-3)=0;
                Rprime=Rpinv';
                Cinv=(Rprime/obj.parammat.Cmat)*Rprime';
                Ejmat=obj.parammat.Ejmat;
                for nmode=1:length(Cinv)
                    if ~any(Lmat(nmode,:))
                        periodic=1;
                        island=1;
                        for rowEJ=1:length(Cinv)
                            for columnEJ=rowEJ:length(Cinv)
                                if columnEJ==rowEJ
                                    exponent=Rpinv(rowEJ,nmode);
                                else
                                    exponent=Rpinv(rowEJ,nmode)-Rpinv(columnEJ,nmode);
                                end
                                if Ejmat(rowEJ,columnEJ)~=0 && exponent~=0
                                    island=0;
                                    if exponent-1>1e-5 && exponent+1>1e-5
                                        expcopy=exponent;
                                        modecopy=nmode;
                                        periodic=0;
                                    end
                                    % A josephon mode is such that the Hamiltonian has
                                    % periodicity Phi0 in its flux
                                    % variable, then it has to appear in
                                    % the Josephson energy terms and the
                                    % coefficients (R)^-1_ik or
                                    % (R)^-1_ik-(R)^-1_jk need to equal
                                    % +or-1
                                end
                            end
                        end
                        if periodic==false
                            fprintf("The rotation specified gives rise to invalid non-H.O. modes: mode %i, exponent=%f\n",modecopy,expcopy);
                        else
                            if island
                                obj.modes(nmode)="Island"; % If Lmat(nmode,nmode)==0 and the Hamiltonian is
                            %not Phi0-periodic in the mode nmode flux, then it is an island mode
                            else
                                obj.modes(nmode)="Jos";
                            end
                        end
                    else
                        obj.modes(nmode)="HO";
                    end
                end
            else
                if abs(det(R))<1e-5
                    fprintf("Error: the R matrix is not invertible\n");
                    return
                end
                Rprime=R;
                Rinv=inv(R);
                Cinv=(R/obj.parammat.Cmat)*R';
                Lmat=(R'\obj.parammat.Lmat)/R;
                Lmat(abs(Lmat)<1e-3)=0;
                Ejmat=obj.parammat.Ejmat;
                for nmode=1:length(Cinv)
                    if ~any(Lmat(nmode,:)) %if the mode does not appear in any inductive term
                        periodic=1; %check if the Hamiltonian is Phi0 periodic in the mode flux
                        island=1;
                        for rowEJ=1:length(Cinv)
                            for columnEJ=rowEJ:length(Cinv)
                                if columnEJ==rowEJ
                                    exponent=Rinv(rowEJ,nmode);
                                else
                                    exponent=Rinv(rowEJ,nmode)-Rinv(columnEJ,nmode);
                                end
                                if Ejmat(rowEJ,columnEJ)~=0 && abs(exponent)>1e-7
                                    island=0;
                                    if exponent-1>1e-5 && exponent+1>1e-5
                                        periodic=0;
                                        expcopy=exponent;
                                        modecopy=nmode;
                                    end
                                    % A josephon mode is such that the Hamiltonian has
                                    % periodicity Phi0 in its flux
                                    % variable, then it has to appear in
                                    % the Josephson energy terms and the
                                    % coefficients (R)^-1_ik or
                                    % (R)^-1_ik-(R)^-1_jk need to equal
                                    % +or-1
                                end
                            end
                        end
                        if periodic==false
                            fprintf("The rotation specified gives rise to invalid non-H.O. modes: mode %i, exponent=%f\n",modecopy,expcopy);
                        else
                            if island
                                obj.modes(nmode)="Island"; % If Lmat(nmode,nmode)==0 and the Hamiltonian is
                            %not Phi0-periodic in the mode nmode flux, then it is an island mode
                            else
                                obj.modes(nmode)="Jos";
                            end
                        end
                    else
                        obj.modes(nmode)="HO";
                    end
                end
            end
        end
        
        function out=incidencemat(obj)
            EJ=obj.parammat.Ejmat;
            nodes=length(obj.modes);
            numJs=length(EJ);
            out=zeros(numJs,nodes);
            EJind=1;
            for ind=find(EJ)'
                [row,col]=ind2sub(nodes,ind);
                out(EJind,row)=-1;
                out(EJind,col)=1;
                EJind=EJind+1;
            end
        end
    end
end
