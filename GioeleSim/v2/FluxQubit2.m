% The FluxQubit class allows the calculation of the eigenspectrum of a
% number of supercondung circuits, as well as other system properties like
% the expectation values of observables, and the coefficeints of the
% underlying qubit Hamiltonian. These properties can also be calculated as 
% a function of a varying physical parameter/magnetic flux. The code is 
% based on the principles of Quantum Circuit Analysis and on the efficient 
% method of representation of the Hamiltonians suggested by J.A. Kerman 
% ("Efficient simulation of complex, Josephson quantum circuits with linear
% inductances", unpublished).

classdef FluxQubit2 < handle
    properties (Constant, Access=private)
        Phi0=2.067834e-15; % Physical constants
        h=6.62607004e-34;
        e=1.60217662e-19;
    end
    properties
        circuitname % Name of the cirucit
        parameters % Physical parameters
        fluxb % fluxbiases
        voltb % voltage biases
       %currb % current biases 
        sweeps % contains the parameters sweeps defied for the object
        modes % list of circuit modes
        truncations % list of truncations associated with each mode
        oplist % list of current and charge operators available for the circuit
        rotation % rotation between the circuit graph node representation and the new modes representation
        subdivision % used to group modes into subsystems
%     end
%     properties (Access={?QubitsInteract})
        ioperators % current operators coordinates
        closures % coordinates of the closure branches
        inductors % list of inductor positions
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
        function obj=FluxQubit2(name)
            if nargin==0
                fprintf('Specify the circuit name.\n')
            elseif nargin==1
                obj.parammatext={};
                switch(name)
                    case "fluxRF"
                        obj.circuitname=name;
                        obj.parameters=struct('Lq',4.5e-9,'Ic',2e-7,'Csh',4.5e-14,'Sc',60,'Jc',3);
                        obj.fluxb=struct('fz',0);
                        obj.voltb={};
                        obj.closures=struct('positions',{[1,1]},'names',{'fz'});
                        obj.inductors=struct('positions',{[1,1]},'names',{'Lq'});
                        obj.modes={"HO"};
                        obj.truncations=50;
                        obj.oplist={'Iz','Q1'};
                        obj.rotation=1;
                        obj.ioperators=struct('Iz',{{{1,0},(@(x) -x)}});
                        obj.subdivision={};
                    case "fluxRF2loops"
                        obj.circuitname=name;
                        obj.parameters=struct('Lc',1.2e-9,'Lco1',1e-11,'Lco2',1e-11,'Ic1',... 
                            1.4e-7,'Ic2',1.4e-7,'Csh',2e-14,'Sc',60,'Jc',3);
                        obj.fluxb=struct('fz',0,'fx',0);
                        obj.voltb={};
                        obj.closures=struct('positions',{[1 2],[1 3]},'names',{{'fz','fx',(@(p) p(1)-p(2)./2)},{'fz','fx',(@(p) p(1)+p(2)./2)}});
                        obj.inductors=struct('positions',{[1,1],[2,2],[3,3]},'names',{'Lc','Lco1','Lco2'});
                        obj.modes={"HO","HO","HO"};
                        obj.truncations=[30, 5, 5];
                        obj.oplist={'Iz','Ix','Q1','Q2','Q3'};
                        obj.rotation=[[1,-0.5,-0.5];[0,0.5,0.5];[0,-1,1]];
                        obj.ioperators=struct('Iz',{{{1,0},(@(x) -x)}},'Ix',{{{2,0},{3,0},(@(x,y) (y-x)./2)}});
                        obj.subdivision={};
                    case "flux3JJ"
                        obj.circuitname=name;
                        obj.parameters=struct('Ic',5.1e-7,'alpha',0.8,'Csh',2e-14,'Sc',60,'Jc',3);
                        obj.fluxb=struct('fz',0);
                        obj.voltb={};
                        obj.closures=struct('positions',{[1,2]},'names',{'fz'});
                        obj.modes={"Jos","Jos"};
                        obj.truncations=[10 10];
                        obj.oplist={'Iz','Q1','Q2'};
                        obj.ioperators=struct('Iz',{{2,1}});
                        obj.subdivision={};
                    case "flux3JJL"
                        obj.circuitname=name;
                        obj.parameters=struct('Ic',5.1e-7,'alpha',0.8,'Csh',2e-14,'Lq',2e-10,'Sc',60,'Jc',3);
                        obj.fluxb=struct('fz',0);
                        obj.voltb={};
                        obj.closures=struct('positions',{[2,3]},'names',{'fz'});
                        obj.inductors=struct('positions',{[1,1]},'names',{'Lq'});
                        obj.modes={"HO","Jos","Jos"};
                        obj.truncations=[7 7 7];
                        obj.oplist={'Iz','Q1','Q2','Q3'};
                        obj.rotation=eye(3);
                        obj.ioperators=struct('Iz',{{1,0}});
                        obj.subdivision={};
                    case "flux4JJL"
                        obj.circuitname=name;
                        obj.parameters=struct('Ic',0.22098e-6,'alpha',0.92316,'Csh',4e-14,'Lq',...
                            2.5e-10,'Lco1',1e-11,'Lco2',1e-11,'Sc',60,'Jc',3);
                        obj.fluxb=struct('fz',0,'fx',0);
                        obj.voltb={};
                        obj.closures=struct('positions',{[1 4],[1 5]},'names',{{'fz','fx',(@(p) p(1)-p(2)./2)},{'fz','fx',(@(p) p(1)+p(2)./2)}});
                        obj.inductors=struct('positions',{[2,2],[3,4],[3,5]},'names',{'Lq','Lco1','Lco2'});
                        obj.modes={"Jos","Jos","HO","HO","HO"};
                        obj.truncations=[6 6 6 4 3];
                        obj.oplist={'Iz','Ix','Q1','Q2','Q3','Q4','Q5'};
                        obj.rotation=[[0,0,-1/3,-1/3,-1/3];[1,0,0,0,0];[0,1,0,0,0];[0,0,0,-1,1];[0,0,-1,0.5,0.5]];
                        obj.ioperators=struct('Iz',{{2,0}},'Ix',{{{4,3},{5,3},(@(x,y) (y-x)./2)}});
                        obj.subdivision={};
                    case "JPSQRF"
                        obj.circuitname=name;
                        obj.parameters=struct('Icl',1.2e-7,'Icr',1.2e-7,'Cl',1e-15,'Cr',1e-15,'Csh',...
                            1e-15,'Ll',7.5e-10,'Lr',7.5e-10,'Sc',60,'Jc',3);
                        obj.fluxb=struct('fz',0);
                        obj.voltb={};
                        obj.closures=struct('positions',{[1,2]},'names',{'fz'});
                        obj.inductors=struct('positions',{[1,1],[3,3]},'names',{'Ll','Lr'});
                        obj.modes={"HO","HO","Jos"};
                        obj.truncations=[5,4,8];
                        obj.oplist={'Iz','Q1','Q2','Q3'};
                        obj.rotation=[[1,0,-1];[1/2,0,1/2];[0,1,0]];
                        %obj.ioperators=struct('Iz',{{{3,0},(@(x) -x)}});
                        obj.ioperators=struct('Iz',{{1,0}});
                        obj.subdivision={};
                        obj.setvoltb(2,[5e-15,0.5]);
                    case "JPSQRF2loops"
                        obj.circuitname=name;
                        obj.parameters=struct('IclT',1.32e-7,'IclB',1.32e-7,'IcrT',1.32e-7,'IcrB',1.32e-7,'Cl',0.1e-15,...
                            'Cr',0.1e-15,'Csh',0,'Cshl',0,'Cshr',0,'Ll',7.5e-10,'Lr',7.5e-10,'LlT',5e-11,'LlB',5e-11,...
                            'LrT',5e-11,'LrB',5e-11,'Sc',60,'Jc',3);
                        obj.fluxb=struct('fz',0,'fL',0,'fR',0);
                        obj.voltb={};
                        obj.closures=struct('positions',{[2,3],[2,4],[2,5]},'names',{{'fz','fL','fR',(@(p) -p(1)+(p(2)+p(3))./2)},{'fz','fL','fR',(@(p) -p(1)+(p(3)-p(2))./2)},'fR'});
                        obj.inductors=struct('positions',{[1,1],[7,7],[1,3],[1,4],[5,7],[6,7]},'names',{'Ll','Lr','LlT','LlB','LrT','LrB'});
                        obj.modes={"HO","HO","HO","HO","HO","HO","Jos"};
                        obj.truncations=[6,4,3,2,2,4,5];
                        obj.oplist={'Iz','Il','Ir','Q1','Q2','Q3','Q4','Q5','Q6','Q7'};
                        obj.rotation=[[1/4,0,-1/8,-1/8,1/8,1/8,-1/4];[0,0,1/4,1/4,1/4,1/4,0];[0,0,1/2,1/2,-1/2,-1/2,0];...
                            [0,0,1,-1,0,0,0];[0,0,0,0,-1,1,0];[1/2,0,0,0,0,0,1/2];[0,1,-1/4,-1/4,-1/4,-1/4,0]];
                        obj.ioperators=struct('Iz',{{1,0}},'Il',{{{3,1},{4,1},(@(x,y) -(x-y)./2)}},'Ir',{{{5,7},{6,7},(@(x,y) (y-x)./2)}});
                        obj.subdivision={{{3,4,5,7},100};{{1,2,6},30}};
                        obj.setvoltb(2,[5e-15,0.5]);
                    case "LCres"
                        obj.circuitname=name;
                        obj.parameters=struct('C',7.8e-13,'L',1.95e-9);
                        obj.fluxb={};
                        obj.voltb={};
                        obj.closures={};
                        obj.inductors=struct('positions',{[1,1]},'names',{'L'});
                        obj.modes={"HO"};
                        obj.truncations=20;
                        obj.oplist={'Q1','I'};
                        obj.rotation=1;
                        obj.ioperators=struct('I',{{1,0}});
                        obj.subdivision={};
                    case "squidres"
                        obj.circuitname=name;
                        obj.parameters=struct('Lc',1.37e-10,'Lco',1e-11,'Ic1',1e-6,'Ic2',1e-6,...
                            'Csh',1e-15,'wr',7,'Zr',50,'Sc',60,'Jc',3);
                        obj.fluxb=struct('fz',0,'fx',0);
                        obj.voltb={};
                        obj.closures=struct('positions',{[1 2],[1 3]},'names',{'f12','f13'}); %f12=fz, f13=fz+fx
                        obj.modes={"HO","HO","HO","HO"};
                        obj.truncations=[7 7 5 5];
                        obj.oplist={'Iz','Ix','Q1','Q2','Q3','Q4'};
                        obj.rotation=[[1,0,0,0];[-1,0,0,1];[0,1,0,0];[0,0,1,0]];
                        obj.ioperators=struct('Iz',{{{1,0},(@(x) -x)}},'Ix',{{{2,0},{3,0},(@(x,y) (y-x)./2)}});
                        obj.subdivision={};
                    otherwise
                        fprintf('Use a valid circuit name\n');
                end
            end
        end
        
        %Get methods for dependent properties:
        
        %% Calculates internal capacitances and Josephson Energies (from Ic)
        function a=get.extraparams(obj)
            if obj.circuitname=="fluxRF"
                a=struct('CJ',obj.parameters.Sc/obj.parameters.Jc*obj.parameters.Ic*10^(-9),'EJ',...
                    obj.h*obj.parameters.Ic/(4*pi*obj.e));
            elseif obj.circuitname=="flux3JJ"
                a=struct('CJ',obj.parameters.Sc/obj.parameters.Jc*obj.parameters.Ic*10^(-9),'CJa',...
                    obj.parameters.Sc/obj.parameters.Jc*obj.parameters.Ic*10^(-9)*obj.parameters.alpha,...
                    'EJ',obj.h*obj.parameters.Ic/(4*pi*obj.e),'EJa',...
                    obj.h*obj.parameters.Ic/(4*pi*obj.e)*obj.parameters.alpha);
            elseif obj.circuitname=="fluxRF2loops"
                a=struct('CJ1',obj.parameters.Sc/obj.parameters.Jc*obj.parameters.Ic1*10^(-9),...
                    'CJ2',obj.parameters.Sc/obj.parameters.Jc*obj.parameters.Ic1*10^(-9),...
                    'EJ1',obj.h*obj.parameters.Ic1/(4*pi*obj.e),'EJ2',obj.h*obj.parameters.Ic2/(4*pi*obj.e));
            elseif obj.circuitname=="flux3JJL"
                a=struct('CJ',obj.parameters.Sc/obj.parameters.Jc*obj.parameters.Ic*10^(-9),'CJa',...
                    obj.parameters.Sc/obj.parameters.Jc*obj.parameters.Ic*10^(-9)*obj.parameters.alpha,...
                    'EJ',obj.h*obj.parameters.Ic/(4*pi*obj.e),'EJa',...
                    obj.h*obj.parameters.Ic/(4*pi*obj.e)*obj.parameters.alpha);
            elseif obj.circuitname=="flux4JJL"
                a=struct('CJ',obj.parameters.Sc/obj.parameters.Jc*obj.parameters.Ic*10^(-9),'CJa',...
                    obj.parameters.Sc/obj.parameters.Jc*obj.parameters.Ic*10^(-9)*obj.parameters.alpha,...
                    'EJ',obj.h*obj.parameters.Ic/(4*pi*obj.e),'EJa',...
                    obj.h*obj.parameters.Ic/(4*pi*obj.e)*obj.parameters.alpha);
            elseif obj.circuitname=="JPSQRF"
                a=struct('CJl',obj.parameters.Sc/obj.parameters.Jc*obj.parameters.Icl*10^(-9),'CJr',...
                    obj.parameters.Sc/obj.parameters.Jc*obj.parameters.Icr*10^(-9),...
                    'EJl',obj.h*obj.parameters.Icl/(4*pi*obj.e),'EJr',...
                    obj.h*obj.parameters.Icr/(4*pi*obj.e));
            elseif obj.circuitname=="JPSQRF2loops"
                a=struct('CJlT',obj.parameters.Sc/obj.parameters.Jc*obj.parameters.IclT*10^(-9),'CJlB',...
                    obj.parameters.Sc/obj.parameters.Jc*obj.parameters.IclB*10^(-9),'CJrT',...
                    obj.parameters.Sc/obj.parameters.Jc*obj.parameters.IcrT*10^(-9),...
                    'CJrB',obj.parameters.Sc/obj.parameters.Jc*obj.parameters.IcrB*10^(-9),'EJlT',...
                    obj.h*obj.parameters.IclT/(4*pi*obj.e),'EJlB',obj.h*obj.parameters.IclB/(4*pi*obj.e),'EJrT',...
                    obj.h*obj.parameters.IcrT/(4*pi*obj.e),'EJrB',obj.h*obj.parameters.IcrB/(4*pi*obj.e));
            elseif obj.circuitname=="LCres"
                a=struct('omegap',1/sqrt(obj.parameters.C*obj.parameters.L)/(2*pi)*1e-9,'Z',...
                    sqrt(obj.parameters.L/obj.parameters.C));
            elseif obj.circuitname=="squidres"
                a=struct('CJ1',obj.parameters.Sc/obj.parameters.Jc*obj.parameters.Ic1*10^(-9),'CJ2',...
                    obj.parameters.Sc/obj.parameters.Jc*obj.parameters.Ic1*10^(-9),...
                    'EJ1',obj.h*obj.parameters.Ic1/(4*pi*obj.e),'EJ2',obj.h*obj.parameters.Ic2/(4*pi*obj.e),...
                    'Lr',obj.parameters.Zr/(2*pi*1e9*obj.parameters.wr),'Cr',(obj.parameters.Zr*2*pi*1e9*obj.parameters.wr)^-1);
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
            if isempty(obj.parammatext) %check if the capacitance and inductive 
                % matrices have been changed by coupling the circuit (i.e.
                % if the values of these matrices have been copied onto
                % parammatext)
                switch(obj.circuitname)
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
                    case "flux3JJL"
                        a=struct('Cmat',[[obj.extraparams.CJ,0,-obj.extraparams.CJ];...
                            [0,obj.extraparams.CJ+obj.extraparams.CJa+obj.parameters.Csh,...
                            -obj.extraparams.CJa-obj.parameters.Csh];[-obj.extraparams.CJ,...
                            -obj.extraparams.CJa-obj.parameters.Csh,obj.extraparams.CJ+obj.extraparams.CJa+...
                            obj.parameters.Csh]],'Lmat',diag([1/obj.parameters.Lq,0,0]),...
                            'Ejmat',[[0,0,obj.extraparams.EJ];[0,obj.extraparams.EJ,obj.extraparams.EJa];[0,0,0]]);
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
                end
                a.Cmat=a.Cmat+obj.gatescap; %Add gates capacitances
            else
                a=obj.parammatext; %Allows the external setting of Cmat and Lmat (eg. to include loading from other circuits)
            end
        end
        
        %% Draws a graph of the circuit
        function circuitgraph(obj)
            switch(obj.circuitname)
                case "fluxRF"
                    G=graph([[0 1];[1 0]],{'n0','n1'});
                    p1=plot(G,'EdgeLabel',{'Lq//Ic//Csh'},'LineWidth',2,'NodeFontSize',16,'EdgeFontSize',16);
                    highlight(p1,[1,2],'EdgeColor','r')
                    set(gca,'visible','off')
                case "fluxRF2loops"
                    G=graph([[0,1,1,1];[1,0,1,1];[1,1,0,0];[1,1,0,0]],{'n0','n1','n2','n3'});
                    p1=plot(G,'EdgeLabel',{'Lc//Csh','Lco1','Lco2','Ic2','Ic1'},'LineWidth',2,'NodeFontSize',16,'EdgeFontSize',16);
                    highlight(p1,[2,3],'EdgeColor','r')
                    highlight(p1,[2,4],'EdgeColor','r')
                    set(gca,'visible','off')
                case "flux3JJ"
                    G=graph([[0,1,1];[1,0,1];[1,1,0]],{'n0','n1','n2'});
                    p1=plot(G,'EdgeLabel',{'Ic','Ic','Ica//Csh'},'LineWidth',2,'NodeFontSize',16,'EdgeFontSize',16);
                    highlight(p1,[2,3],'EdgeColor','r')
                    set(gca,'visible','off')
                case "flux3JJL"
                    G=graph([[0,1,1,0];[1,0,0,1];[1,0,0,1];[0,1,1,0]],{'n0','n1','n2','n3'});
                    p1=plot(G,'EdgeLabel',{'Lc','Ic','Ic','Ica//Csh'},'LineWidth',2,'NodeFontSize',16,'EdgeFontSize',16);
                    highlight(p1,[3,4],'EdgeColor','r')
                    set(gca,'visible','off')
                case "flux4JJL"
                    G=graph([[0,1,1,0,0,0];[1,0,0,1,1,1];[1,0,0,1,0,0];[0,1,1,0,1,1];[0,1,0,1,0,0];[0,1,0,1,0,0]],{'n0','n1','n2','n3','n4','n5'});
                    p1=plot(G,'EdgeLabel',{'Ic','Lq','Csh','Ica','Ica','Ic','Lco1','Lco2'},'LineWidth',2,'Layout','layered','Sources',[1 2 6] ,'Sinks',[3 4],'NodeFontSize',16,'EdgeFontSize',16);
                    highlight(p1,[2,5],'EdgeColor','r')
                    highlight(p1,[2,6],'EdgeColor','r')
                    set(gca,'visible','off')
                case "JPSQRF"
                    G=graph([[0,1,1,1];[1,0,1,1];[1,1,0,1];[1,1,1,0]],{'n0','n1','n2','n3'});
                    p1=plot(G,'EdgeLabel',{'Cl||Ll','Cg','Cr||Lr','Icl','Csh','Icr'},'LineWidth',2,'NodeFontSize',16,'EdgeFontSize',16);
                    highlight(p1,[2,3],'EdgeColor','r')
                    set(gca,'visible','off')
                case "LCres"
                    G=graph([[0 1];[1 0]],{'n0','n1'});
                    plot(G,'EdgeLabel',{'L//C'},'LineWidth',2,'NodeFontSize',16,'EdgeFontSize',16);
                    set(gca,'visible','off')
                otherwise
                    fprintf("Circuit graph not specified for this circuit\n");
            end
        end
        
        %% Calculate the contibution to the capacitance matrix due to the gates capacitors
        function a=get.gatescap(obj)
            a=zeros(1,length(obj.rotation));
            for ind=1:length(obj.voltb) % define the diagonal elements as a vector
                a(obj.voltb(ind).node)=obj.voltb(ind).Cg;
            end
            a=diag(a); % convert the vector into the diagonal matrix
        end
        
        %Set methods
        
%         %% Rotation set method: updates the modes according to the new to rotation matrix R
%         function set.rotation(obj,R)
%             Rprime=obj.updatemodes(R); % check that the rotation is valid and determine the types of the new modes
%             obj.rotation=Rprime; % then replace the rotation property
%         end
        
        %% Set a voltage bias by specifying node number, gate capacitance and gate
        % voltage (in units of the Cooper pair charge 2e)
        function setvoltb(obj,node,vec)
            if isempty(obj.voltb)
                obj.voltb=struct('node',node,'Cg',vec(1),'Qg',vec(2));
            else
                obj.voltb(end+1)=struct('node',node,'Cg',vec(1),'Qg',vec(2));
            end
        end
        
        % Core functions:
        
%         %% Calculates plasma frequencies for all modes
%         % required for the terms hbar*omega_p*(N+1/2) associated with each
%         % oscillator mode
%         function out=getPlasmaf(obj)
%             R=obj.rotation;
%             Cinv=(R/obj.parammat.Cmat)*R'; %capacitive and inductive matrices in the rotated representation
%             Lmat=(R'\obj.parammat.Lmat)/R;
%             Lmat(abs(Lmat)<1e-3)=0; %neglect tiny terms
%             Cdiag=diag(Cinv);
%             Ldiag=diag(Lmat);
%             fpsquared=Cdiag.*Ldiag;
%             out=arrayfun(@sqrt,fpsquared)'; % omega_p=1/sqrt(LC)
%         end
        
        %% Generates bias matrix, where M(i,j)=exp(i*phi^ext_ij) if (i,j) 
        % is a closure branch, M(i,j)=1 otherwise
        function out=getBiasm(obj,fluxbiases) % the optional input is used in flux sweeps 
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
                else
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
                end
            end
        end
        
        %% Displays eigenfunctions for the rf-SQUID qubit
        % using the expressions for the wavefuction of the n-th occupation
        % number eigenstate, which contains a gaussian factor and the n-th
        % Hermite polynomial. The function returns the list of values of
        % the eigenfunction for values of phase in the range (-pi,pi), as
        % well as the expectation value phi of the phase operator
        function [out,phi]=wavefunc(obj,state)
            if isequal(obj.circuitname,"fluxRF")
                fz=obj.fluxb.fz*2*pi;
                f=@(n,x) hermiteH(n,x)./sqrt(2^n*factorial(n));
                H=obj.getH;
                [P,~]=obj.getPQ;
                p0=sqrt(2)/P(1,2)*obj.Phi0/(2*pi); % numerical factor appearing in the Gaussian term of the wavefunction 
                [evect,evals]=eigs(H,state,'sr'); % find the eigenstates of the Hamiltonian
                [~,order]=sort(real(diag(evals))); % check that the eigenvalues are in the right order
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
                set(gca,'fontsize',16);
                xlabel('$\phi/2\pi$','FontSize',18,'Interpreter','latex')
                ylab=strcat('$\Psi_',num2str(state),'(\phi)$');
                leglab=strcat('$|\Psi_',num2str(state),'(\phi)|^2$');
                ylabel(ylab,'FontSize',18,'Interpreter','latex')
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
        function [P,Q]=getPQ(obj,smallout)
            nummodes=length(obj.modes); % number of modes
            truncation=obj.truncations;
            R=obj.rotation;
            Cinv=(R/obj.parammat.Cmat)*R'; %capacitive and inductive matrices in the rotated representation
            Lmat=(R'\obj.parammat.Lmat)/R;
            for ntrunc=1:length(truncation)
                if (obj.modes{ntrunc}=="Jos") || (obj.modes{ntrunc}=="Island")
                    truncation(ntrunc)=2*truncation(ntrunc)+1; %2N+1 states for Josephson and Island modes,
                    % makes sure that the identity matrices for every mode
                    % have the right diemensions
                end
            end
            Id=cellfun(@speye,num2cell(truncation),'UniformOutput',0); %cell of identity matrices with the appropriate dimension for each mode.
            %Used to calculate kronecker products
            if nargin<2 || (nargin==2 && smallout==false)
                % allocate 3d sparse matrices filled with zeros
                P=ndSparse.spalloc([prod(truncation) prod(truncation) nummodes],2*nummodes*prod(truncation)); % Flux and charge operators for every mode
                Q=ndSparse.spalloc([prod(truncation) prod(truncation) nummodes],2*nummodes*prod(truncation));
            elseif nargin==2 && smallout==true
                P=Id;
                Q=Id;
            end
            for modenum=1:nummodes
                if obj.modes{modenum}=="HO"
                    % for harmonic oscillator modes, express P and Q in
                    % terms of ladder operators. The matrices are truncated
                    % at a given occupation number
                    Psmall=Id;
                    Qsmall=Id;
                    diagonal=sqrt(1:truncation(modenum));
                    diagM=spdiags(diagonal',0,truncation(modenum),truncation(modenum)); %sparse diagonal matrix
                    a=circshift(diagM,1,2);
                    a(:,1)=0;
                    adagger=a';
                    Z=sqrt(Cinv(modenum,modenum)/Lmat(modenum,modenum));
                    Psmall(1,modenum)={sqrt(obj.h*Z/(4*pi))*(a+adagger)};
                    Qsmall(1,modenum)={1i*sqrt(obj.h/(4*pi*Z))*(adagger-a)};
                    if nargin<2 || (nargin==2 && smallout==false)
                        % expressions of the P and Q operators in the
                        % composite Hilbert space obtained by taking the
                        % outer product of each mode Hilbert space
                        P(:,:,modenum)=superkron(Psmall);
                        Q(:,:,modenum)=superkron(Qsmall);
                    elseif nargin==2 && smallout==true
                        % expressions of the P and Q operators in the
                        % single mode Hilbert spaces
                        P(1,modenum)=Psmall(1,modenum);
                        Q(1,modenum)=Qsmall(1,modenum);
                    end
                elseif (obj.modes{modenum}=="Jos") || (obj.modes{modenum}=="Island")
                    % For Josephson and island modes, only the charge
                    % operators are necessary. These are expressed in the
                    % charge basis, hence they are diagonal. Matrices are
                    % truncated at a given absolute charge (in integer
                    % multiples of 2e: Q=diag(-2Ne,...,2Ne))
                    Qsmall=Id;
                    diagM=spdiags(fliplr(-obj.truncations(modenum):obj.truncations(modenum))',0,truncation(modenum),truncation(modenum));
                    Qsmall(1,modenum)={2*obj.e*diagM};
                    if nargin<2 || (nargin==2 && smallout==false)
                        Q(:,:,modenum)=superkron(Qsmall);
                    elseif nargin==2 && smallout==true
                        Q(1,modenum)=Qsmall(1,modenum);
                    end
                end
            end
        end
        
        %% Calculates Displacement operators for every branch with a JJ
        % these represent the term exp(i*phi_ab) of the cosine appearing in
        % the Josephson energy relative to the junction on the branch ab.
        % phi_ab=2*pi/Phi_0*(Phi_a-Phi_b), where Phi_i is the node flux
        % associated with the node i.
        function Dmat=getDops(obj,smallout)
            truncation=obj.truncations;
            for ntrunc=1:length(truncation)
                if (obj.modes{ntrunc}=="Jos") || (obj.modes{ntrunc}=="Island")
                    truncation(ntrunc)=2*truncation(ntrunc)+1; %2N+1 states for Josephson and Island modes
                    % makes sure that the identity matrices for every mode
                    % have the right diemensions
                end
            end
            nummodes=length(obj.modes);
            R=obj.rotation;
            Rinv=inv(R);
            Rinvt=Rinv';
            EJ=obj.parammat.Ejmat; % matrix containing the josephson energies of the junctions on each branch
                                   % connecting two given nodes
            [P,~]=obj.getPQ(true);
            D=cell(1,nummodes);
            Dmat=cell(nummodes,nummodes);
            exponents=zeros(nummodes,nummodes,nummodes);
            Id=cellfun(@speye,num2cell(truncation),'UniformOutput',0);
            for modenum=1:nummodes
                exponents(:,:,modenum)=Rinv(:,modenum)-Rinvt(modenum,:); %exponent appearing in front of the mode flux operator
                exponents=exponents+diag(Rinv(:,modenum)); %diagonal terms
                if obj.modes{modenum}=="Jos"
                    Dp=circshift(speye(truncation(modenum)),-1,2);
                    Dp(:,end)=0;
                    D(1,modenum)={Dp};
                end
            end
                % D(i,j)==exp(1i.*(P_i-P_j))=exp(1i.*sum_k((R)^-1_ik-(R)^-1_jk)P'_k),
                % and D(i,i)==exp(1i.*P_i)=exp(1i.*sum_k((R)^-1_ik*P'_k))=prod_k(exp(1i.*(R)^-1_ik*P'_k)),
                % where the P_k's are the original modes (corresponding to
                % the circuit graph) and the P'_k's are the ones after the
                % rotation R
            for ind=find(EJ)' %for every nonzero element in the Josephson energy matrix ==
                % for every branch containing a Josephson junction
                [Dmatrow,Dmatcolumn]=ind2sub(size(EJ,1),ind);
                Dcopy=Id;
                for modenum=1:nummodes
                    if exponents(Dmatrow,Dmatcolumn,modenum)~=0
                        if obj.modes{modenum}=="HO"
                            Dcopy(1,modenum)={expm(1i*exponents(Dmatrow,Dmatcolumn,modenum).*2*pi/obj.Phi0.*P{1,modenum})};
                        elseif obj.modes{modenum}=="Jos"
                            if exponents(Dmatrow,Dmatcolumn,modenum)==1
                                Dcopy(1,modenum)=D(1,modenum);
                            elseif exponents(Dmatrow,Dmatcolumn,modenum)==-1
                                Dcopy(1,modenum)={D{1,modenum}'};
                            end
                        end
                    end
                end
                if ~smallout
                    Dmat(Dmatrow,Dmatcolumn)={superkron(Dcopy)};
                else
                    Dmat(Dmatrow,Dmatcolumn)=Dcopy;
                end
            end
        end
        
        %% Computes capacitive and inductive energy Hamiltonian
        % subsoption toggles between the calculation of the subsystems
        % energies and the interaction energies for a partitioned system
        function out=getHLC(obj,biastoggle,subsoption,Cinv,Lmat,P,Q)
            if nargin<2
                biastoggle=false; %leave out the voltage bias terms for voltage sweeps
            elseif (nargin<3 && isempty(obj.subdivision)) || (isequal(subsoption,'interaction') && nargin==6)
                if (nargin<3 && isempty(obj.subdivision)) % if no partitioning is present
                    R=obj.rotation;
                    Cinv=(R/obj.parammat.Cmat)*R'; %capacitive and inductive matrices in the rotated representation
                    Lmat=(R'\obj.parammat.Lmat)/R;
                    [P,Q]=obj.getPQ(true); % flux and charge operators
                    truncation=obj.truncations;
                    for ntrunc=1:length(truncation)
                        if (obj.modes{ntrunc}=="Jos") || (obj.modes{ntrunc}=="Island")
                            truncation(ntrunc)=2*truncation(ntrunc)+1; %2N+1 states for Josephson and Island modes
                            % makes sure that the identity matrices for every mode
                            % have the right diemensions
                        end
                    end
                    Id=cellfun(@speye,num2cell(truncation),'UniformOutput',0); %cell of identity matrices of edge truncation(modenum)
                    out=ndSparse.spalloc([prod(truncation) prod(truncation)],2*prod(truncation));
                    for row=1:length(Cinv)
                        for col=row:length(Cinv)
                            if row==col %diagonal terms
                                if Cinv(row,row)~=0
                                    Qcopy=Id;
                                    Qcopy(1,row)={Q{1,row}^2};
                                    out=out+0.5*Cinv(row,row).*superkron(Qcopy);
                                end
                                if Lmat(row,row)~=0
                                    Pcopy=Id;
                                    Pcopy(1,row)={P{1,row}^2};
                                    out=out+0.5*Lmat(row,row).*superkron(Pcopy);
                                end
                            else %off-diagonal terms
                                if Cinv(row,col)~=0
                                    Qcopy=Id;
                                    Qcopy(1,col)=Q(1,col);
                                    Qcopy(1,row)=Q(1,row);
                                    out=out+Cinv(row,col).*superkron(Qcopy); 
                                end
                                if Lmat(row,col)~=0
                                    Pcopy=Id;
                                    Pcopy(1,col)=P(1,col);
                                    Pcopy(1,row)=P(1,row);
                                    out=out+Lmat(row,col).*superkron(Pcopy);
                                end
                            end
                        end
                    end
                    if ~isempty(obj.voltb) && ~biastoggle % Add the voltage bias term
                        % express the charge operator in the outer product basis:
                        Q2=ndSparse.spalloc([prod(truncation) prod(truncation) length(R)],2*length(R)*prod(truncation));
                        for mode=1:length(R)
                            Qcopy=Id;
                            Qcopy(1,mode)=Q(1,mode);
                            Q2(:,:,mode)=superkron(Qcopy);
                        end
                        Qoff=zeros(1,length(R)); % get voltage offsets
                        for ind=1:length(obj.voltb)
                            Qoff(obj.voltb(ind).node)=obj.voltb(ind).Qg*2*obj.e;
                        end
                        Qoff=(inv(R)'*Qoff')'; % voltage offsets in the modes representation
                        Hextra=ndSparse.spalloc([prod(truncation) prod(truncation)],2*prod(truncation));
                        if any(Qoff)
                            for ind=find(Qoff)
                                Hextra=Hextra+Qoff(ind).*ncon({Cinv(ind,:)',Q2},{1,[-1 -2 1]})+...
                                    Cinv(ind,ind)*(Qoff(ind))^2/2.*eye(length(out));
                            end
                        end
                        out=out+Hextra;
                    end
                    out=out.'./obj.h*1e-9; % transpose the result and convert to GHz units
                elseif (isequal(subsoption,'interaction') && nargin==6) % calculate interaction terms between
                    % different modes using the provided flux and charge
                    % operators
                    out=ndSparse.spalloc([size(P,1) size(P,1)],2*size(P,1));
                    for row=1:length(Cinv)
                        for col=row:length(Cinv)
                            if Cinv(row,col)~=0
                                out=out+Cinv(row,col).*Q(:,:,row)*Q(:,:,col); 
                            end
                            if Lmat(row,col)~=0
                                out=out+Lmat(row,col).*P(:,:,row)*P(:,:,col);
                            end
                        end
                    end
                    if ~isempty(obj.voltb) && ~biastoggle %Add the voltage bias term
                        % express the charge operator in the outer product basis:
                        R=obj.rotation;
                        Qoff=zeros(1,length(R)); % get voltage offsets
                        for ind=1:length(obj.voltb)
                            Qoff(obj.voltb(ind).node)=obj.voltb(ind).Qg*2*obj.e;
                        end
                        Qoff=(inv(R)'*Qoff')'; % voltage offsets in the modes representation
                        Hextra=ndSparse.spalloc([length(out) length(out)],2*length(out));
                        if any(Qoff)
                            for ind=find(Qoff)
                                Hextra=Hextra+Qoff(ind).*ncon({Cinv(ind,:)',Q},{1,[-1 -2 1]})+...
                                    Cinv(ind,ind)*(Qoff(ind))^2/2.*eye(length(out));
                            end
                        end
                        out=out+Hextra;
                    end
                    out=out./obj.h*1e-9; % convert to GHz units
                end
            elseif nargin==3 && isequal(subsoption,'self')
                out=cell(1,length(obj.subdivision));
                R=obj.rotation;
                Cinv=(R/obj.parammat.Cmat)*R'; %capacitive and inductive matrices in the rotated representation
                Lmat=(R'\obj.parammat.Lmat)/R;
                [P,Q]=obj.getPQ(true); % flux and charge operators
                truncation=obj.truncations;
                for ntrunc=1:length(truncation)
                    if (obj.modes{ntrunc}=="Jos") || (obj.modes{ntrunc}=="Island")
                        truncation(ntrunc)=2*truncation(ntrunc)+1; %2N+1 states for Josephson and Island modes
                        % makes sure that the identity matrices for every mode
                        % have the right diemensions
                    end
                end
                for subind=1:length(obj.subdivision) %for every subsystem
                    Id=cellfun(@speye,obj.subdivision{subind}{1},'UniformOutput',0); %cell of identity matrices of edge truncation(modenum)
                    
                end
            end
            
        end
        
        %% Build operators in the subsystem basis.
        % Given a set of operators for each mode "op" and a set 
        % on each circuit in the rapresentation defined in the level 1 of
        % truncation, this function returns the expression of the
        % Hamiltonian term in the 2nd order truncation basis
        function Hnew=lev2truncate(obj,op,U)
            numqubits=length(obj.circuits); %number or circuits in the system
            if length(H)==numqubits %check that operators are given for every circuit
                Hnewtemp=cell(1,length(U)); %one term for every subsystem
                for subind=1:length(obj.lev2truncation) %for every lower subsystem
                    circs=cell2mat(obj.lev2truncation{subind,:}{1}); %circuits in subsystem subind
                    subsize=length(circs); %number of circuits in the subsystem
                    truncs=obj.lev1truncation(circs); %corresponding truncations
                    Id=cellfun(@eye,num2cell(truncs),'UniformOutput',0);
                    %Hcopy=zeros(prod(truncs));
                    %Hn=Id;
                    Hcopy=Id;
                    for circn=1:subsize
                        Hcopy(1,circn)=H(1,circs(circn)); %copy the operator for circuit circs(circn)
                    end
                    Hcopy=superkron(Hcopy);
                    %rotate to the new basis
                    Hnewtemp(1,subind)={ncon({conj(U{1,subind}),Hcopy,U{1,subind}},{[1 -1],[1 2],[2,-2]})};
                end
                if ~isempty(obj.lev3truncation)
                    Hnewtemp2=cell(1,length(obj.lev3truncation)); %one term for every higher subsystem
                    truncs=(cellfun(@(x) x{2},obj.lev2truncation))';
                    for subind=1:length(obj.lev3truncation) %for every lowe subsystem
                        circs=cell2mat(obj.lev3truncation{subind,:}{1}); %circuits in subsystem subind
                        subsize=length(circs); %number of circuits in the subsystem
                        truncsi=truncs(circs); %corresponding truncations
                        Id=cellfun(@eye,num2cell(truncsi),'UniformOutput',0);
                        %Hcopy=zeros(prod(truncs));
                        %Hn=Id;
                        Hcopy=Id;
                        for circn=1:subsize
                            Hcopy(1,circn)=Hnewtemp(1,circs(circn)); %copy the operator for circuit circs(circn)
                        end
                        Hcopy=superkron(Hcopy);
                        %rotate to the new basis
                        Hnewtemp2(1,subind)={ncon({conj(U{1,length(obj.lev2truncation)+... 
                            subind}),Hcopy,U{1,length(obj.lev2truncation)+subind}},{[1 -1],[1 2],[2,-2]})};
                    end
                end
                %take outer product of the different terms
                if ~isempty(obj.lev3truncation)
                    Hnew=superkron(Hnewtemp2);
                else
                    Hnew=superkron(Hnewtemp);
                end
            else
               fprintf("Incorrect structure of subsystem Hamiltonians\n") 
            end
        end
        
%         %% Calculates N operators
%         % these are the number operators used to write the terms
%         % hbar*omega_p*(N+1/2) associated with each oscillator mode
%         function N=getNops(obj)
%             truncation=obj.truncations;
%             for ntrunc=1:length(truncation)
%                 if (obj.modes{ntrunc}=="Jos") || (obj.modes{ntrunc}=="Island")
%                     truncation(ntrunc)=2*truncation(ntrunc)+1; %2N+1 states for Josephson and Island modes
%                     % makes sure that the identity matrices for every mode
%                     % have the right diemensions
%                 end
%             end
%             nummodes=length(obj.modes);
%             % allocate 3d sparse matrices filled with zeros
%             N=ndSparse.spalloc([prod(truncation) prod(truncation) nummodes],nummodes*prod(truncation));
%             Id=arrayfun(@speye,truncation,'UniformOutput',0);% cell of identity matrices with the appropriate dimension for each mode.
%             %Used to calculate kronecker products
%             for modenum=1:nummodes
%                 if obj.modes{modenum}=="HO" %Number operators only appear for oscillator modes
%                     Nsmall=Id;
%                     Nsmall(1,modenum)={spdiags((0:truncation(modenum)-1)',0,truncation(modenum),truncation(modenum))}; % diagon matrix with integer eigenvalues from 0 to Nmax
%                     N(:,:,modenum)=superkron(Nsmall);
%                 end
%             end
%         end
        
        % Sweep definition methods:
        
        %% Paramsweep method: defines a parameter sweep
        function paramsweep(obj,sweepname,param,range)
            % PARAMSWEEP Defines a parameter sweep.
            %
            %   obj.paramsweep("name",'param',r) defines a sweep with name
            %   "name", which sweeps the parameter 'param' in the range r
            %   (where r is an array). Multiple parameters can be sweeped
            %   at the same time by using a cell of strings for the
            %   parameter names and a cell of arrays for the corresponding
            %   ranges
            if iscell(param) % if more then one parameter is given in a cell
                toggle=0;
                for paramn=1:length(param)
                    if isfield(obj.parameters,param{paramn}) && isempty(obj.sweeps) % check that the parameter name is valid
                        % and if the list of sweeps is still empty
                        obj.sweeps=struct('sweepname',sweepname,'sweeptype','param','parameter',param{paramn},'range',range{paramn});
                    elseif isfield(obj.parameters,param{paramn})
                        % if the list is not empty
                        if ~sum(cellfun(@(x,y) isequal(x,y),{obj.sweeps.sweepname},repmat({sweepname},1,length(obj.sweeps)))) || toggle
                            % first check that no sweep with the same name
                            % exists already, then add more elements to the
                            % sweeps structure
                            obj.sweeps(end+1)=struct('sweepname',sweepname,'sweeptype','param','parameter',param{paramn},'range',range{paramn});
                        else
                            fprintf('Sweep name already in use\n');
                        end
                    else
                        fprintf('Use a valid param sweep parameter\n');
                    end
                    toggle=toggle+1; % skip the check of duplicate sweep names for successive parameters
                end
            else % if only one parameter is given
                if isfield(obj.parameters,param) && isempty(obj.sweeps) % check that the parameter name is valid
                    % and if the list of sweeps is still empty
                    obj.sweeps=struct('sweepname',sweepname,'sweeptype','param','parameter',param,'range',range);
                elseif isfield(obj.parameters,param)
                    % if the list is not empty
                    if ~sum(cellfun(@(x,y) isequal(x,y),{obj.sweeps.sweepname},repmat({sweepname},1,length(obj.sweeps))))
                        % check that no sweep with the same name exists already, then add the
                        % new element to the sweeps structure
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
            %   (where r is an array). Multiple fluxes can be sweeped
            %   at the same time by using a cell of strings for the
            %   flux names and a cell of arrays for the corresponding
            %   ranges
            if isempty(obj.closures)
                fprintf("This function is only available for circuits with at least one irreducible loop.\n")
            else
                if iscell(bias) % if more then one flux is given in a cell
                    toggle=0;
                    for biasn=1:length(bias)
                        if isfield(obj.fluxb,bias{biasn}) && isempty(obj.sweeps) % check that the flux name is valid
                            % and if the list of sweeps is still empty
                            obj.sweeps=struct('sweepname',sweepname,'sweeptype','flux','parameter',bias{biasn},'range',range{biasn});
                        elseif isfield(obj.fluxb,bias{biasn})
                            % if the list is not empty
                            if ~sum(cellfun(@(x,y) isequal(x,y),{obj.sweeps.sweepname},repmat({sweepname},1,length(obj.sweeps)))) || toggle
                                % first check that no sweep with the same name
                                % exists already, then add more elements to the
                                % sweeps structure
                                obj.sweeps(end+1)=struct('sweepname',sweepname,'sweeptype','flux','parameter',bias{biasn},'range',range{biasn});
                            else
                                fprintf('Sweep name already in use\n');
                            end
                        else
                            fprintf('Use a valid flux sweep parameter\n');
                        end
                        toggle=toggle+1; % skip the check of duplicate sweep names for successive fluxes
                    end
                else % if only one flux is given
                    if isfield(obj.fluxb,bias) && isempty(obj.sweeps) % check that the flux name is valid
                        % and if the list of sweeps is still empty
                        obj.sweeps=struct('sweepname',sweepname,'sweeptype','flux','parameter',bias,'range',range);
                    elseif isfield(obj.fluxb,bias)
                        % if the list is not empty
                        if ~sum(cellfun(@(x,y) isequal(x,y),{obj.sweeps.sweepname},repmat({sweepname},1,length(obj.sweeps))))
                            % check that no sweep with the same name exists already, then add the
                            % new element to the sweeps structure
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
        
        %% Voltage sweep method: defines a flux bias sweep
        function voltsweep(obj,sweepname,bias,range)
            % VOLTSWEEP Defines a voltage bias sweep.
            %
            %   obj.voltsweep("name",'bias',r) defines a sweep with name
            %   "name", which sweeps the voltage applied to the node 'bias' 
            %   in the range r (where r is an array). Multiple voltages can 
            %   be sweeped at the same time by using a cell for the
            %   node numbers and a cell of arrays for the corresponding
            %   ranges
            if isempty(obj.voltb)
                fprintf("Set the voltage bias, including gate capacitance, first.\n")
            else
                if iscell(bias) % if more then one voltage is given in a cell
                    toggle=0;
                    for biasn=1:length(bias)
                        if ~isempty(find([obj.voltb.node]==bias{biasn}, 1)) && isempty(obj.sweeps) % check that a voltage 
                            % bias, including the gate capacitance, has
                            % been previously defined for this node and
                            % check if the sweeps list is empty
                            obj.sweeps=struct('sweepname',sweepname,'sweeptype','voltage','parameter',bias{biasn},'range',range{biasn});
                        elseif ~isempty(find([obj.voltb.node]==bias{biasn}, 1))
                            % if the list is not empty
                            if ~sum(cellfun(@(x,y) isequal(x,y),{obj.sweeps.sweepname},repmat({sweepname},1,length(obj.sweeps)))) || toggle
                                % check that no sweep with the same name exists already, then add the
                                % new element to the sweeps structure
                                obj.sweeps(end+1)=struct('sweepname',sweepname,'sweeptype','voltage','parameter',bias{biasn},'range',range{biasn});
                            else
                                fprintf('Sweep name already in use\n');
                            end
                        else
                            fprintf('Define a voltage bias, including a gate capacitance, for this node first.\n');
                        end
                        toggle=toggle+1; % skip the check of duplicate sweep names for successive voltages
                    end
                else % if only one voltage is given
                    if ~isempty(find([obj.voltb.node]==bias, 1)) && isempty(obj.sweeps) % check that a voltage 
                            % bias, including the gate capacitance, has
                            % been previously defined for this node and
                            % check if the sweeps list is empty
                        obj.sweeps=struct('sweepname',sweepname,'sweeptype','voltage','parameter',bias,'range',range);
                    elseif ~isempty(find([obj.voltb.node]==bias, 1))
                        % if the list is not empty
                        if ~sum(cellfun(@(x,y) isequal(x,y),{obj.sweeps.sweepname},repmat({sweepname},1,length(obj.sweeps))))
                            % check that no sweep with the same name exists already, then add the
                            % new element to the sweeps structure
                            obj.sweeps(end+1)=struct('sweepname',sweepname,'sweeptype','voltage','parameter',bias,'range',range);
                        else
                            fprintf('Sweep name already in use\n');
                        end
                    else
                        fprintf('Define a voltage bias, including a gate capacitance, for this node first.\n');
                    end
                end
            end
        end
        
        %% Mixed flux and voltage sweep method: defines a flux and voltage bias sweep
        function mixedsweep(obj,sweepname,bias,range)
            % MIXEDSWEEP Defines a mixed flux/voltage bias sweep.
            %
            %   obj.mixedsweep("name",'bias',r) defines a sweep with name
            %   "name", which sweeps the voltages and the fluxes with names 
            %   in the cell list 'bias' in the range r (where r is an array).
            %   range is a list with the arrays r.
            if isempty(obj.voltb)
                fprintf("Set the voltage bias, including gate capacitance, first.\n")
            elseif isempty(obj.closures)
                fprintf("This function is only available for circuits with at least one irreducible loop.\n")
            else
                toggle=0;
                for biasn=1:length(bias)
                    if ischar(bias{biasn}) % is the bias name is the name of a flux bias
                        if isfield(obj.fluxb,bias{biasn}) && isempty(obj.sweeps) % check that the flux name is valid
                            % and if the list of sweeps is still empty
                            obj.sweeps=struct('sweepname',sweepname,'sweeptype','mixed','parameter',bias{biasn},'range',range{biasn});
                        elseif isfield(obj.fluxb,bias{biasn})
                            % if the list is not empty
                            if ~sum(cellfun(@(x,y) isequal(x,y),{obj.sweeps.sweepname},repmat({sweepname},1,length(obj.sweeps)))) || toggle
                                % first check that no sweep with the same name
                                % exists already, then add more elements to the
                                % sweeps structure
                                obj.sweeps(end+1)=struct('sweepname',sweepname,'sweeptype','mixed','parameter',bias{biasn},'range',range{biasn});
                            else
                                fprintf('Sweep name already in use\n');
                            end
                        else
                            fprintf('Use a valid flux sweep parameter\n');
                        end
                        toggle=toggle+1; % skip the check of duplicate sweep names for successive voltages
                    else
                        if ~isempty(find([obj.voltb.node]==bias{biasn}, 1)) && isempty(obj.sweeps) % check that a voltage 
                            % bias, including the gate capacitance, has
                            % been previously defined for this node and
                            % check if the sweeps list is empty
                            obj.sweeps=struct('sweepname',sweepname,'sweeptype','mixed','parameter',bias{biasn},'range',range{biasn});
                        elseif ~isempty(find([obj.voltb.node]==bias{biasn}, 1))
                            % if the list is not empty
                            if ~sum(cellfun(@(x,y) isequal(x,y),{obj.sweeps.sweepname},repmat({sweepname},1,length(obj.sweeps)))) || toggle
                                % check that no sweep with the same name exists already, then add the
                                % new element to the sweeps structure
                                obj.sweeps(end+1)=struct('sweepname',sweepname,'sweeptype','mixed','parameter',bias{biasn},'range',range{biasn});
                            else
                                fprintf('Sweep name already in use\n');
                            end
                        else
                            fprintf('Define a voltage bias, including a gate capacitance, for this node first.\n');
                        end
                        toggle=toggle+1; % skip the check of duplicate sweep names for successive voltages
                    end
                end
            end
        end
    end
end