% The QubitsInteract class allows to establish capacitive and inductive
% interactions between FluxQubits objects and to calculate the energy
% spectrum of the obtained composite systems.

classdef QubitsInteract < handle
    properties (Constant, Access=private)
        Phi0=2.067834e-15; %Physical constants
        h=6.62607004e-34;
        e=1.60217662e-19;
        Sc=60;%50; %Fabrication constants
        Jc=3;%2.78;
    end
    properties
        circuits
        circuitnames
        interactions
        sweeps
        lev1truncation
        lev2truncation
        lev3truncation
    end
    methods
        
        %% Constructor
        function obj=QubitsInteract(varargin)
            if nargin==0
                fprintf("Input the interacting circuits as a FluxQubit objects.\n")
            else
                obj.circuitnames=cell(1,nargin);
                obj.circuits=varargin{1};
                obj.circuitnames(1,1)={inputname(1)};
                for circuitn=2:nargin
                    obj.circuits(end+1)=varargin{circuitn};
                    obj.circuitnames(1,circuitn)={inputname(circuitn)};
                end
            end
        end
        
        %% Checks that the right number of truncation values is assigned
        function set.lev1truncation(obj,v)
            if length(v)==length(obj.circuitnames)
                obj.lev1truncation=v;
            else
                fprintf("Error: %i truncation values required\n",length(obj.circuitnames));
            end
        end
        
        %% Calculates the energy spectrum of the coupled system for a fixed parameter configuration
        % CALCULATEE Calculates the energy spectrum of the system given
        % its current parameters and biases.
        %
        %   out=obj.calculateE(levelsout) produces the first levelsout energy
        %   levels of the system.
        %   out=obj.calculateE(obj,levelsout,'Relative',option) subtracts
        %   the ground state energy from every level if option is set
        %   to true.
        function [out,Htot]=calculateE(obj,outlevels,relative,option)
            numqubits=length(obj.circuitnames);
            if length(obj.interactions)<1
                fprintf("Define pairs of interacting qubits\n");
%             elseif length(obj.minductances)+length(obj.couplingcap)<length(obj.interpairs)
%                 fprintf("Define mutual inductances/shared capacitances values\n");
%             elseif length(obj.coupledinductors)+length(obj.couplednodes)<length(obj.interpairs)
%                 fprintf("Define coupling branch currents or node charges\n");
            elseif length(obj.lev1truncation)<numqubits
                fprintf("Define upper level truncation values\n");
            else
                if any(cellfun(@(x) isequal(x,'Ind'),{obj.interactions.ctype})) %If an inductive coupling is present
                    [dict,Lmatg]=obj.buildinductancemat(); %Get global branch inductance matrix
                    Lmatg=inv(Lmatg); %invert the inductance matrix
                else
                    Lmatg=[];
                end
                if any(cellfun(@(x) isequal(x,'Cap'),{obj.interactions.ctype})) %If a capacitive coupling is present
                    [~,Cmatg,sizes]=obj.buildcapacitancemat(); %Get inverse global node capacitance matrix
                else
                    Cmatg=[];
                end
                obj.setloadparammats; %Change capacitance and inductance matrices to account for capative/inductive loads
                numinter=length(find(triu(Lmatg,1)))+length(find(triu(Cmatg,1))); %Number of interaction terms
                %it equals half the number of off diagonal terms in the
                %inductance matrix plus half the number of block-off diagonal
                %terms in the capacitance matrix 
                if isempty(obj.lev2truncation) %if no second level truncation is defined
                    Hbig=zeros(prod(obj.lev1truncation)); %Sum of the single system hamiltonians
                    Hint=zeros(prod(obj.lev1truncation),prod(obj.lev1truncation),numinter); %Interaction hamiltonians
                else
                    if isempty(obj.lev3truncation)
                        dims=obj.lev2truncation;
                    else
                        dims=obj.lev3truncation;
                    end
                    subsnum=size(dims,1);
                    for subind=1:subsnum
                        if subind==1
                            sizetot=dims{subind,:}{2};
                        else
                            sizetot=sizetot*dims{subind,:}{2};
                        end
                    end
                    Hint=zeros(sizetot,sizetot,numinter);
                end
                Id=cellfun(@eye,num2cell(obj.lev1truncation),'UniformOutput',0);
                Evect=cell(1,numqubits); %Cell of eigenvectors
                H0lev2=cell(1,numqubits); %Unperturbed Hamiltonian for 2nd level hierarchical diagonalisation
                
                %Unperturbed Hamiltonians eigenvalues problem
                for nqubit=1:numqubits
                    Hsmall=obj.circuits(nqubit).getH;
                    [Evect{1,nqubit},evals]=eigs(sparse(Hsmall),obj.lev1truncation(nqubit),'sr');
                    Htemp=Id;
                    Htemp(1,nqubit)={real(evals)};
                    if ~isempty(obj.lev2truncation)
                        H0lev2(1,nqubit)={real(evals)}; %copy single systems H0 for 2nd level h. diagonalisation
                    else
                        Hbig=Hbig+superkron(Htemp); %if no 2nd order h.d. is required, calculate the total H0
                        %i.e. the sum of the single circuits unperturbed
                        %Hamiltonians (corrected for capacitive/inductive
                        %loading)
                    end
                end
                if ~isempty(obj.lev2truncation)
                    U=obj.lev2rotations(H0lev2); %calculate rotation mapping to the subsystems basis
                    Hbig=zeros(sizetot);
                    for nqubit=1:numqubits
                        Htemp=Id;
                        Htemp(1,nqubit)=H0lev2(1,nqubit);
                        Hbig=Hbig+obj.lev2truncate(Htemp,U); %apply rotation
                    end
                end
                intindex=1;
                if ~isempty(Lmatg)
    %                 Hint=zeros(prod(obj.lev1truncation),prod(obj.lev1truncation),intnum);
                    for Lind=find(triu(Lmatg,1))'%Lmatrow=1:length(Lmatg)
                        [Lmatrow,Lmatcol]=ind2sub(size(Lmatg),Lind);
                        %for Lmatcol=Lmatrow+1:length(Lmatg)
                        %for every element of the inductance matrix,
                        %find the involved circuits
                        circn1=dict(Lmatrow).circuit;
                        circn2=dict(Lmatcol).circuit;
                        v1=Evect{1,circn1}; %get the unperturbed eigenvectors
                        v2=Evect{1,circn2};
                        %get the position of the two inductors to
                        %calculate the branch fluxes
                        inductn1=cellfun(@(x) isequal(x,dict(Lmatrow).inductor),{obj.circuits(circn1).inductors.names});
                        positions1=obj.circuits(dict(Lmatrow).circuit).inductors(inductn1).positions;
                        inductn2=cellfun(@(x) isequal(x,dict(Lmatcol).inductor),{obj.circuits(circn2).inductors.names});
                        positions2=obj.circuits(dict(Lmatcol).circuit).inductors(inductn2).positions;
                        [P1,~]=obj.circuits(circn1).getPQ;
                        [P2,~]=obj.circuits(circn2).getPQ;
                        Rinv1=inv(obj.circuits(circn1).rotation);
                        Rinv2=inv(obj.circuits(circn2).rotation);
                        if positions1(1)==positions1(2)
                            Pb1=squeeze(ncon({Rinv1(positions1(1),:),P1},{[-1,1],[-2 -3 1]}));
                        else
                            Pb1=squeeze(ncon({Rinv1(positions1(2),:),P1},{[-1,1],[-2 -3 1]}))-squeeze(ncon({Rinv1(positions1(1),:),P1},{[-1,1],[-2 -3 1]}));
                        end
                        if positions2(1)==positions2(2)
                            Pb2=squeeze(ncon({Rinv2(positions2(1),:),P2},{[-1,1],[-2 -3 1]}));
                        else
                            Pb2=squeeze(ncon({Rinv2(positions2(2),:),P2},{[-1,1],[-2 -3 1]}))-squeeze(ncon({Rinv2(positions2(1),:),P2},{[-1,1],[-2 -3 1]}));
                        end
                        %project the branch fluxes on the unperturbed
                        %Hamiltonian eigenvectors basis
                        H1=ncon({conj(v1),Pb1,v1},{[1 -1],[1 2],[2,-2]}); %Hint=H1*H2
                        H2=ncon({conj(v2),Pb2,v2},{[1 -1],[1 2],[2,-2]});
%                         H1=ncon({v1,Pb1,conj(v1)},{[1 -1],[1 2],[2,-2]}); %Hint=H1*H2
%                         H2=ncon({v2,Pb2,conj(v2)},{[1 -1],[1 2],[2,-2]});
                        Hintern=Id;
                        if circn1==circn2
                            Hintern{1,circn1}=H1*H2; %mediated self-interaction term
                        else
                            Hintern{1,circn1}=H1;
                            Hintern{1,circn2}=H2;
                        end
                        %every interachtion term is of the form
                            %(L^-1)_ij*Phi_i*Phi_j with i!=j
                        if ~isempty(obj.lev2truncation)
                            Hinttemp=obj.lev2truncate(Hintern,U); %map interaction terms to subsystem basis
                            Hint(:,:,intindex)=Lmatg(Lmatrow,Lmatcol).*Hinttemp./obj.h*1e-9;
                            intindex=intindex+1;
                            %fprintf(strcat('intindex=',num2str(intindex-1),', norm(H)=',num2str(norm(Hint(:,:,intindex-1)),'%e'),'\n'))
                        else
                            Hint(:,:,intindex)=Lmatg(Lmatrow,Lmatcol)*superkron(Hintern)./obj.h*1e-9;
                            intindex=intindex+1;
                            %fprintf(strcat('intindex=',num2str(intindex),', norm(H)=',num2str(norm(Hint(:,:,intindex)))))
                        end
                        %end
                    end
                end
                if ~isempty(Cmatg) && ~isempty(find(Cmatg,1))
                    for ind=find(triu(Cmatg,1))'
                        [node1,node2]=ind2sub([length(Cmatg),length(Cmatg)],ind);
                        circuitmarker=cumsum(sizes); %cumulative sum of the sequence of sizes of the capacitance
                        %matrices
                        circn1=find(node1<=circuitmarker,1);
                        circn2=find(node2<=circuitmarker,1);
                        v1=Evect{1,circn1}; %get the unperturbed eigenvectors
                        v2=Evect{1,circn2};
                        if circn1>1 %find the node number in the subcircuit
                            node1=node1-(circuitmarker(circn1-1));
                        end
                        if circn2>1
                            node2=node2-(circuitmarker(circn2-1));
                        end
                        [~,Q1]=obj.circuits(circn1).getPQ(false,true);
                        [~,Q2]=obj.circuits(circn2).getPQ(false,true);
                        Rinv1=obj.circuits(circn1).rotation';
                        Rinv2=obj.circuits(circn2).rotation';
                        Q1=squeeze(ncon({Rinv1(node1,:),Q1},{[-1,1],[-2 -3 1]}));
                        Q2=squeeze(ncon({Rinv2(node2,:),Q2},{[-1,1],[-2 -3 1]}));
                        %project the currents on the unperturbed
                        %Hamiltonian eigenvectors basis
                        H1=ncon({conj(v1),Q1,v1},{[1 -1],[1 2],[2,-2]}); %Hint=H1*H2
                        H2=ncon({conj(v2),Q2,v2},{[1 -1],[1 2],[2,-2]});
%                         H1=ncon({v1,Q1,conj(v1)},{[1 -1],[1 2],[2,-2]}); %Hint=H1*H2
%                         H2=ncon({v2,Q2,conj(v2)},{[1 -1],[1 2],[2,-2]});
                        Hintern=Id;
                        if circn1==circn2
                            Hintern{1,circn1}=H1*H2; %mediated self-interaction term
                        else
                            Hintern{1,circn1}=H1;
                            Hintern{1,circn2}=H2;
                        end
                        %every interaction term is of the form
                            %(C^-1)_ij*Q_i*Q_j with i!=j
                        if ~isempty(obj.lev2truncation) %map interaction terms to subsystem basis
                            Hinttemp=obj.lev2truncate(Hintern,U);
                            Hint(:,:,intindex)=Cmatg(ind).*Hinttemp./obj.h*1e-9;
                            intindex=intindex+1;
                        else
                            Hint(:,:,intindex)=Cmatg(ind)*superkron(Hintern)./obj.h*1e-9;
                            intindex=intindex+1;
                        end
                    end
                end
                Htot=Hbig+sum(Hint,3);
                opts.maxit=500;
                eigenvals=eigs(sparse(Htot),outlevels,'sr',opts);
                out=sort(real(eigenvals));
                out=out(1:outlevels);
                if nargin==4 && isequal(relative,'Relative') && option==true %subtract ground state energy
                    out=out-out(1,1);
                end
                obj.resetloadparammats; %reset capacitive and inductive matrices
            end
        end
        
        %% Calculates the energy spectrum of the coupled system as a function of one or more varying parameters
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
        function out=sweepE(obj,sweepnm,outlevels,relative,optionr,plotopt,optionp)
            numqubits=length(obj.circuitnames);
            if length(obj.interactions)<1
                fprintf("Define pairs of interacting qubits\n");
%             elseif length(obj.minductances)+length(obj.couplingcap)<length(obj.interpairs)
%                 fprintf("Define mutual inductances/shared capacitances values\n");
%             elseif length(obj.coupledinductors)+length(obj.couplednodes)<length(obj.interpairs)
%                 fprintf("Define coupling branch currents or node charges\n");
            elseif length(obj.lev1truncation)<numqubits
                fprintf("Define upper level truncation values\n");
            elseif nargin < 3
                fprintf("Specify sweepname and number of output levels\n");
            elseif isempty(obj.sweeps)
                fprintf("Define a sweep first using the function paramsweep or fluxsweep\n");
            else
                if any(cellfun(@(x) isequal(x,'Ind'),{obj.interactions.ctype}))
                    [dict,Lmatg]=obj.buildinductancemat(); %get global branch inductance matrix
                    Lmatg=inv(Lmatg); %invert the inductance matrix
                else
                    Lmatg=[];
                end
                if any(cellfun(@(x) isequal(x,'Cap'),{obj.interactions.ctype}))
                    [~,Cmatg,sizes]=obj.buildcapacitancemat(); %get inverse global node capacitance matrix
                else
                    Cmatg=[];
                end
                obj.setloadparammats;
                numinter=length(find(triu(Lmatg,1)))+length(find(triu(Cmatg,1))); %Number of interaction terms
                %it equals half the number of off diagonal terms in the
                %inductance matrix plus half the number of block-off diagonal
                %terms in the capacitance matrix 
                H1=cell(1,numinter);
                H2=cell(1,numinter);
                numsweptpar=0;
                toggle=1;
                %check that the specified sweep has been previously defined
                %(through paramsweep() or fluxsweep())
                for sweepn=1:length(obj.sweeps)
                    if isequal(obj.sweeps(sweepn).sweepname,sweepnm)
                        numsweptpar=numsweptpar+1; %if a sweep with the desired name is found, increase numsweptpar by 1
%                         if strcmp(obj.sweeps(sweepn).parameter(1),'L') || strcmp(obj.sweeps(sweepn).parameter(1),'C') || isequal(obj.sweeps(sweepn).parameter,'alpha') 
%                             %if the sweep changes the the flux/current operators, turn toggle on,
%                             toggle=1;
%                         end
                    end
                end
                if numsweptpar==0
                    fprintf("Specify an existent sweep name\n");
                else
                    %copy the sweep data
                    sweep=struct('sweepname',sweepnm,'qubitnum',cell(1,numsweptpar),'sweeptype',cell(1,numsweptpar),'parameter',...
                        cell(1,numsweptpar),'range',cell(1,numsweptpar));
                    param=1;
                    for sweepn=1:length(obj.sweeps)
                        if isequal(obj.sweeps(sweepn).sweepname,sweepnm)
                            sweep(param)=obj.sweeps(sweepn); % Only copy the sweeps with the right name
                            param=param+1;
                        end
                    end
                    sweeplengths=cellfun(@length,{sweep.range});
                    if all(sweeplengths==sweeplengths(1)) %checks that all swept parameters take the same number of values
                        out=zeros(length(sweep(1).range),outlevels);
                        if isequal(sweep(1).sweeptype,'param') %parameter sweep: requires calculating the full hamiltonian every time
                            paramold=zeros(1,numsweptpar); %copy the original value of the swept parameter
                            for paramn=1:numsweptpar
                                paramold(1,paramn)=obj.circuits(sweep(paramn).qubitnum).parameters.(sweep(paramn).parameter);
                            end
                            f=waitbar(1/length(sweep(1).range),strcat('Sweep progress:',num2str(100/length(sweep(1).range),2),'%'));
                            for iter=1:length(sweep(1).range) %for every point in the sweep
                                waitbar(iter/length(sweep(1).range),f,strcat('Sweep progress:',num2str(100*iter/length(sweep(1).range),2),'%'));
                                for paramn=1:numsweptpar %change qubit properties
                                    obj.circuits(sweep(paramn).qubitnum).parameters.(sweep(paramn).parameter)=sweep(paramn).range(iter);
                                end
                                if toggle %recalculate loaded matrices to account for a change in L or C
                                    obj.resetloadparammats;
                                    if any(cellfun(@(x) isequal(x,'Ind'),{obj.interactions.ctype}))
                                        [dict,Lmatg]=obj.buildinductancemat(); %get global branch inductance matrix
                                        Lmatg=inv(Lmatg); %invert the inductance matrix
                                    else
                                        Lmatg=[];
                                    end
                                    if any(cellfun(@(x) isequal(x,'Cap'),{obj.interactions.ctype}))
                                        [~,Cmatg,sizes]=obj.buildcapacitancemat(); %get inverse global node capacitance matrix
                                    else
                                        Cmatg=[];
                                    end
                                    obj.setloadparammats;
                                end
%                                 if iter==1
                                    %initialise cells at the first
                                    %iteration
                                    if isempty(obj.lev2truncation)
                                        Hbig=zeros(prod(obj.lev1truncation)); %Sum of the single system hamiltonians
                                        Hint=zeros(prod(obj.lev1truncation),prod(obj.lev1truncation),numinter); %Interaction hamiltonians
                                    else
                                        if isempty(obj.lev3truncation)
                                            dims=obj.lev2truncation;
                                        else
                                            dims=obj.lev3truncation;
                                        end
                                        subsnum=size(dims,1);
                                        for subind=1:subsnum
                                            if subind==1
                                                sizetot=dims{subind,:}{2};
                                            else
                                                sizetot=sizetot*dims{subind,:}{2};
                                            end
                                        end
                                        Hint=zeros(sizetot,sizetot,numinter);
                                    end
                                    Evect=cell(1,numqubits);
                                    H0lev2=cell(1,numqubits); %Unperturbed Hamiltonian for 2nd level hierarchical diagonalisation
                                    Id=cellfun(@eye,num2cell(obj.lev1truncation),'UniformOutput',0);
                                    P=cell(1,numqubits);
                                    Q=cell(1,numqubits);
                                    %calculate operators necessary for
                                    %interaction terms
                                    if ~isempty(Lmatg) %if there are iductive couplings
                                        for nqubit=1:numqubits %node fluxes
                                            [Ptemp,~]=obj.circuits(nqubit).getPQ;
                                            Rinv=inv(obj.circuits(nqubit).rotation);
                                            P(1,nqubit)={ncon({Rinv,Ptemp},{[-3,1],[-1 -2 1]})};
                                        end
                                    end
                                    if ~isempty(Cmatg) && ~isempty(find(Cmatg,1))
                                        for nqubit=1:numqubits %node charges
                                            [~,Qtemp]=obj.circuits(nqubit).getPQ;
                                            Rinv=obj.circuits(nqubit).rotation';
                                            Q(1,nqubit)={ncon({Rinv,Qtemp},{[-3,1],[-1 -2 1]})};
                                        end
                                    end
                                    %solve unperturbed Hamiltonian problems
                                    for nqubit=1:numqubits
                                        Hsmall=obj.circuits(nqubit).getH;
%                                         Htemp=Id;
                                        [Evect{1,nqubit},evals]=eigs(sparse(Hsmall),obj.lev1truncation(nqubit),'SR');
%                                         [Evect{1,nqubit},Htemp{1,nqubit}]=eigs(Hsmall,obj.lev1truncation(nqubit),'SR');
%                                         [evals,order]=sort(real(diag(evals)));
%                                         Evect(1,nqubit)={Evecttemp(:,order)};
                                        Htemp=Id;
                                        Htemp(1,nqubit)={real(evals)};
                                        if ~isempty(obj.lev2truncation)
                                            H0lev2(1,nqubit)={real(evals)}; %copy single systems H0 for 2nd level h. diagonalisation
                                        else
                                            Hbig=Hbig+superkron(Htemp); %if no 2nd order h.d. is required, calculate the total H0
                                        end
                                    end
                                    if ~isempty(obj.lev2truncation)
                                        U=obj.lev2rotations(H0lev2);
                                        %                     Hbig=obj.lev2truncate(H0lev2,U);
                                        Hbig=zeros(sizetot);
                                        for nqubit=1:numqubits
                                            Htemp=Id;
                                            Htemp(1,nqubit)=H0lev2(1,nqubit);
                                            Hbig=Hbig+obj.lev2truncate(Htemp,U);
                                        end
                                    end
                                    %find the interaction terms
                                    intindex=1;
                                    if ~isempty(Lmatg)
                                        for Lind=find(triu(Lmatg,1))'%Lmatrow=1:length(Lmatg)
                                            [Lmatrow,Lmatcol]=ind2sub(size(Lmatg),Lind);%for Lmatcol=Lmatrow+1:length(Lmatg)
                                            %for every element of the inductance matrix,
                                            %find the involved circuits
                                            circn1=dict(Lmatrow).circuit;
                                            circn2=dict(Lmatcol).circuit;

                                            v1=Evect{1,circn1}; %get the unperturbed eigenvectors
                                            inductn1=cellfun(@(x) isequal(x,dict(Lmatrow).inductor),{obj.circuits(circn1).inductors.names});
                                            positions1=obj.circuits(dict(Lmatrow).circuit).inductors(inductn1).positions;
                                            if positions1(1)==positions1(2)
                                                Pb1=P{circn1}(:,:,positions1(1));
                                            else
                                                Pb1=P{circn1}(:,:,positions1(2))-P{circn1}(:,:,positions1(1));
                                            end
                                            H1(intindex)={ncon({conj(v1),Pb1,v1},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                            %H1(intindex)={ncon({v1,Pb1,conj(v1)},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2

                                            v2=Evect{1,circn2}; %get the unperturbed eigenvectors
                                            inductn2=cellfun(@(x) isequal(x,dict(Lmatcol).inductor),{obj.circuits(circn2).inductors.names});
                                            positions2=obj.circuits(dict(Lmatcol).circuit).inductors(inductn2).positions;
                                            if positions2(1)==positions2(2)
                                                Pb2=P{circn2}(:,:,positions2(1));
                                            else
                                                Pb2=P{circn2}(:,:,positions2(2))-P{circn2}(:,:,positions2(1));
                                            end
                                            H2(intindex)={ncon({conj(v2),Pb2,v2},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                            %H2(intindex)={ncon({v2,Pb2,conj(v2)},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2

                                            Hintern=Id;
                                            if circn1==circn2
                                                Hintern{1,circn1}=H1{intindex}*H2{intindex}; %mediated self-interaction term
                                            else
                                                Hintern{1,circn1}=H1{intindex};
                                                Hintern{1,circn2}=H2{intindex};
                                            end
%                                             Hintern{1,circn1}=H1{intindex};
%                                             Hintern{1,circn2}=H2{intindex};
                                            if ~isempty(obj.lev2truncation)
                                                Hinttemp=obj.lev2truncate(Hintern,U);
                                                Hint(:,:,intindex)=Lmatg(Lmatrow,Lmatcol).*Hinttemp./obj.h*1e-9;
                                                intindex=intindex+1;
                                            else
                                                %every interachtion term is of the form
                                                %(L^-1)_ij*Phi_i*Phi_j with i!=j
                                                Hint(:,:,intindex)=Lmatg(Lmatrow,Lmatcol)*superkron(Hintern)./obj.h*1e-9;
                                                intindex=intindex+1;
                                            end
                                            %end
                                        end
                                    end
                                    if ~isempty(Cmatg) && ~isempty(find(Cmatg,1))
                                        for ind=find(triu(Cmatg,1))'
                                            %for every element of the inverse matrix,
                                            %find the involved circuits
                                            [node1,node2]=ind2sub([length(Cmatg),length(Cmatg)],ind);
                                            circuitmarker=cumsum(sizes); %cumulative sum of the sequence of sizes of the capacitance
                                            %matrices
                                            circn1=find(node1<=circuitmarker,1);
                                            circn2=find(node2<=circuitmarker,1);
                                            if circn1>1
                                                node1=node1-(circuitmarker(circn1-1));
                                            end
                                            if circn2>1
                                                node2=node2-(circuitmarker(circn2-1));
                                            end
                                            
                                            v1=Evect{1,circn1}; %get the unperturbed eigenvectors
                                            Q1=Q{circn1}(:,:,node1);
                                            H1(intindex)={ncon({conj(v1),Q1,v1},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                            %H1(intindex)={ncon({v1,Q1,conj(v1)},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2

                                            v2=Evect{1,circn2}; %get the unperturbed eigenvectors
                                            Q2=Q{circn2}(:,:,node2);
                                            H2(intindex)={ncon({conj(v2),Q2,v2},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                            %H2(intindex)={ncon({v2,Q2,conj(v2)},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                            
                                            Hintern=Id;
                                            if circn1==circn2
                                                Hintern{1,circn1}=H1{intindex}*H2{intindex}; %mediated self-interaction term
                                            else
                                                Hintern{1,circn1}=H1{intindex};
                                                Hintern{1,circn2}=H2{intindex};
                                            end
%                                             Hintern{1,circn1}=H1{intindex};
%                                             Hintern{1,circn2}=H2{intindex};
                                            if ~isempty(obj.lev2truncation)
                                                Hinttemp=obj.lev2truncate(Hintern,U);
                                                Hint(:,:,intindex)=Cmatg(ind).*Hinttemp./obj.h*1e-9;
                                                intindex=intindex+1;
                                            else
                                                %every interachtion term is of the form
                                                %(C^-1)_ij*Q_i*Q_j with i!=j
                                                Hint(:,:,intindex)=Cmatg(ind)*superkron(Hintern)./obj.h*1e-9;
                                                intindex=intindex+1;
                                            end
                                        end
                                    end
                                %sum unperturbed and perturbed Hamiltonians
                                Htot=sum(Hbig,3)+sum(Hint,3).';
                                eigenvals=eigs(sparse(Htot),outlevels,'SA');
                                out(iter,:)=sort(real(eigenvals));
                                if nargin>3 && isequal(relative,'Relative') && optionr==true
                                    out(iter,:)=out(iter,:)-out(iter,1);
                                end
                            end
                            obj.resetloadparammats; %restore parammats
                            for paramn=1:numsweptpar %restore original param values
                                obj.circuits(sweep(paramn).qubitnum).parameters.(sweep(paramn).parameter)=paramold(1,paramn);
                            end
                        elseif isequal(sweep(1).sweeptype,'flux') %flux sweep, only the bias matrix needs to be recalculated
                            f=waitbar(1/length(sweep(1).range),strcat('Sweep progress:',num2str(100/length(sweep(1).range),2),'%'));
                            for iter=1:length(sweep(1).range) %for every point in the sweep
                                waitbar(iter/length(sweep(1).range),f,strcat('Sweep progress:',num2str(100*iter/length(sweep(1).range),2),'%'));
                                if iter==1
                                    %initialise cells at the first
                                    %iteration
                                    Id=cellfun(@eye,num2cell(obj.lev1truncation),'UniformOutput',0);
                                    if isempty(obj.lev2truncation)
%                                         Hbig=zeros(prod(obj.lev1truncation)); %Sum of the single system hamiltonians
                                        Hbig=zeros(prod(obj.lev1truncation),prod(obj.lev1truncation),numqubits); %Single system hamiltonians
                                        Hint=zeros(prod(obj.lev1truncation),prod(obj.lev1truncation),numinter); %Interaction hamiltonians
                                    else
                                        if isempty(obj.lev3truncation)
                                            dims=obj.lev2truncation;
                                        else
                                            dims=obj.lev3truncation;
                                        end
                                        subsnum=size(dims,1);
                                        for subind=1:subsnum
                                            if subind==1
                                                sizetot=dims{subind,:}{2};
                                            else
                                                sizetot=sizetot*dims{subind,:}{2};
                                            end
                                        end
                                        Hint=zeros(sizetot,sizetot,numinter);
                                    end
                                    HLC=cell(1,numqubits);
                                    Dmat=cell(1,numqubits);
                                    Evect=cell(1,numqubits);
                                    H0lev2=cell(1,numqubits); %Unperturbed Hamiltonian for 2nd level hierarchical diagonalisation
%                                     Hint=zeros(prod(obj.lev1truncation),prod(obj.lev1truncation),numinter); %interaction hamiltonians
                                    P=cell(1,numqubits);
                                    Q=cell(1,numqubits);
                                    %calculate operators necessary for
                                    %interaction terms
                                    if ~isempty(Lmatg) %if there are iductive couplings
                                        for nqubit=1:numqubits %node fluxes
                                            [Ptemp,~]=obj.circuits(nqubit).getPQ;
                                            Rinv=inv(obj.circuits(nqubit).rotation);
                                            P(1,nqubit)={ncon({Rinv,Ptemp},{[-3,1],[-1 -2 1]})};
                                        end
                                    end
                                    if ~isempty(Cmatg) && ~isempty(find(Cmatg,1))
                                        for nqubit=1:numqubits %node charges
                                            [~,Qtemp]=obj.circuits(nqubit).getPQ;
                                            Rinv=obj.circuits(nqubit).rotation';
                                            Q(1,nqubit)={ncon({Rinv,Qtemp},{[-3,1],[-1 -2 1]})};
                                        end
                                    end
                                    nummodes=zeros(1,numqubits);
                                    for nqubit=1:numqubits
                                        %unperturbed Hamiltonians
                                        nummodes(nqubit)=length(obj.circuits(nqubit).modes);
                                        HLC(1,nqubit)={obj.circuits(nqubit).getHLC};
                                        Dmat(1,nqubit)={obj.circuits(nqubit).getDops};
                                        EJ=obj.circuits(nqubit).parammat.Ejmat;
                                        HJ=sum(sum(EJ)).*eye(length(HLC{1,nqubit}));
                                        if any(find([sweep.qubitnum]==nqubit))
                                            biasmatrix=obj.circuits(nqubit).getBiasm(sweep([sweep.qubitnum]==nqubit),1); % pass the value from the sweep only to the qubits
                                        else                                                      % affected by the sweep
                                            biasmatrix=obj.circuits(nqubit).getBiasm();
                                        end
                                        for ind=find(EJ)'
                                            [Dmatrow,Dmatcolumn]=ind2sub(size(EJ,1),ind);
%                                         for Dmatrow=1:nummodes(nqubit)
%                                             for Dmatcolumn=Dmatrow:nummodes(nqubit)
%                                                 if EJ(Dmatrow,Dmatcolumn)~=0 %sum only on the branches containing a JJ
                                            HJ=HJ-EJ(Dmatrow,Dmatcolumn)/2.*(biasmatrix(Dmatrow,Dmatcolumn).*Dmat{1,nqubit}{Dmatrow,Dmatcolumn}+...
                                                conj(biasmatrix(Dmatrow,Dmatcolumn)).*Dmat{1,nqubit}{Dmatrow,Dmatcolumn}');
%                                                 end
%                                             end
                                        end
                                        HJ=HJ./obj.h*1e-9;
%                                         Htemp=Id;
                                        [Evect{1,nqubit},evals]=eigs(sparse(HLC{1,nqubit}+HJ),obj.lev1truncation(nqubit),'SR');
%                                         [evals,order]=sort(real(diag(Htemp)));
%                                         Evect(1,nqubit)={Evecttemp(:,order)};
                                        Htemp=Id;
                                        Htemp(1,nqubit)={real(evals)};
%                                         Hbig(:,:,nqubit)=superkron(Htemp);
                                        if ~isempty(obj.lev2truncation)
                                            %H0lev2(1,nqubit)={diag(evals)}; %copy single systems H0 for 2nd level h. diagonalisation
                                            H0lev2(1,nqubit)={real(evals)}; %copy single systems H0 for 2nd level h. diagonalisation
                                        else
                                            Hbig(:,:,nqubit)=superkron(Htemp); %if no 2nd order h.d. is required, calculate the total H0
                                        end
                                    end
                                    if ~isempty(obj.lev2truncation)
                                        U=obj.lev2rotations(H0lev2);
                                        %                     Hbig=obj.lev2truncate(H0lev2,U);
                                        Hbig=zeros(sizetot);
                                        for nqubit=1:numqubits
                                            Htemp=Id;
                                            Htemp(1,nqubit)=H0lev2(1,nqubit);
                                            Hbig=Hbig+obj.lev2truncate(Htemp,U);
                                        end
                                    end
                                else
                                    for nqubit=unique([sweep.qubitnum]) %update only the unperturbed Hamiltonians
                                        %relative to qubits involved in the sweep
                                        EJ=obj.circuits(nqubit).parammat.Ejmat;
                                        HJ=sum(sum(EJ)).*eye(length(HLC{1,nqubit}));
                                        %update the biases values in the
                                        %Josephson terms
                                        biasmatrix=obj.circuits(nqubit).getBiasm(sweep([sweep.qubitnum]==nqubit),iter);
                                        for ind=find(EJ)'
                                            [Dmatrow,Dmatcolumn]=ind2sub(size(EJ,1),ind);
%                                         for Dmatrow=1:nummodes(nqubit)
%                                             for Dmatcolumn=Dmatrow:nummodes(nqubit)
%                                                 if EJ(Dmatrow,Dmatcolumn)~=0 %sum only on the branches containing a JJ
                                            HJ=HJ-EJ(Dmatrow,Dmatcolumn)/2.*(biasmatrix(Dmatrow,Dmatcolumn).*Dmat{1,nqubit}{Dmatrow,Dmatcolumn}+...
                                                conj(biasmatrix(Dmatrow,Dmatcolumn)).*Dmat{1,nqubit}{Dmatrow,Dmatcolumn}');
%                                                 end
%                                             end
                                        end
                                        HJ=HJ./obj.h*1e-9;
%                                         Htemp=Id;
                                        [Evect{1,nqubit},evals]=eigs(sparse(HLC{1,nqubit}+HJ),obj.lev1truncation(nqubit),'SR');
%                                         [Evect{1,nqubit},Htemp{1,nqubit}]=eigs(HLC{1,nqubit}+HJ,obj.lev1truncation(nqubit),'SR');
%                                         [evals,order]=sort(real(diag(Htemp)));
%                                         Evect(1,nqubit)={Evecttemp(:,order)};
                                        Htemp=Id;
                                        Htemp(1,nqubit)={real(evals)};
%                                         Hbig(:,:,nqubit)=superkron(Htemp);
                                        if ~isempty(obj.lev2truncation)
                                            H0lev2(1,nqubit)={real(evals)}; %copy single systems H0 for 2nd level h. diagonalisation
                                        else
                                            Hbig(:,:,nqubit)=superkron(Htemp); %if no 2nd order h.d. is required, calculate the total H0
                                        end
                                    end
                                    if ~isempty(obj.lev2truncation)
                                        U=obj.lev2rotations(H0lev2);
                                        %                     Hbig=obj.lev2truncate(H0lev2,U);
                                        Hbig=zeros(sizetot);
                                        for nqubit=1:numqubits
                                            Htemp=Id;
                                            Htemp(1,nqubit)=H0lev2(1,nqubit);
                                            Hbig=Hbig+obj.lev2truncate(Htemp,U);
                                        end
                                    end
                                end
                                intindex=1;
                                if ~isempty(Lmatg)
                                    for Lind=find(triu(Lmatg,1))'%Lmatrow=1:length(Lmatg)
                                        [Lmatrow,Lmatcol]=ind2sub(size(Lmatg),Lind);%for Lmatcol=Lmatrow+1:length(Lmatg)
                                        %for every element of the inductance matrix,
                                        %find the involved circuits
                                        circn1=dict(Lmatrow).circuit;
                                        circn2=dict(Lmatcol).circuit;
                                        if length(unique([sweep.qubitnum]))>1 %check if the circuits are involved in the sweep
%                                                 toggle1=contains(unique([sweep.qubitnum]),circn1);
%                                                 toggle2=contains(unique([sweep.qubitnum]),circn2);
                                            toggle1=~isempty(find([sweep.qubitnum]==circn1, 1));
                                            toggle2=~isempty(find([sweep.qubitnum]==circn2, 1));
                                        else
                                            toggle1=isequal(unique([sweep.qubitnum]),circn1);
                                            toggle2=isequal(unique([sweep.qubitnum]),circn2);
                                        end
                                        if iter==1 || toggle1 %if so, recalculate expectation values
                                            %on the new eigenstates
                                            v1=Evect{1,circn1}; %get the unperturbed eigenvectors
                                            inductn1=cellfun(@(x) isequal(x,dict(Lmatrow).inductor),{obj.circuits(circn1).inductors.names});
                                            positions1=obj.circuits(dict(Lmatrow).circuit).inductors(inductn1).positions;
                                            if positions1(1)==positions1(2)
                                                Pb1=P{circn1}(:,:,positions1(1));
                                            else
                                                Pb1=P{circn1}(:,:,positions1(2))-P{circn1}(:,:,positions1(1));
                                            end
                                            H1(intindex)={ncon({conj(v1),Pb1,v1},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                            %H1(intindex)={ncon({v1,Pb1,conj(v1)},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                        end
                                        if iter==1 || toggle2
                                            v2=Evect{1,circn2}; %get the unperturbed eigenvectors
                                            inductn2=cellfun(@(x) isequal(x,dict(Lmatcol).inductor),{obj.circuits(circn2).inductors.names});
                                            positions2=obj.circuits(dict(Lmatcol).circuit).inductors(inductn2).positions;
                                            if positions2(1)==positions2(2)
                                                Pb2=P{circn2}(:,:,positions2(1));
                                            else
                                                Pb2=P{circn2}(:,:,positions2(2))-P{circn2}(:,:,positions2(1));
                                            end
                                            H2(intindex)={ncon({conj(v2),Pb2,v2},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                            %H2(intindex)={ncon({v2,Pb2,conj(v2)},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                        end
                                        Hintern=Id;
                                        if circn1==circn2
                                            Hintern{1,circn1}=H1{intindex}*H2{intindex}; %mediated self-interaction term
                                        else
                                            Hintern{1,circn1}=H1{intindex};
                                            Hintern{1,circn2}=H2{intindex};
                                        end
                                        %                                             Hintern{1,circn1}=H1{intindex};
                                        %                                             Hintern{1,circn2}=H2{intindex};
                                        if ~isempty(obj.lev2truncation)
                                            Hinttemp=obj.lev2truncate(Hintern,U);
                                            Hint(:,:,intindex)=Lmatg(Lmatrow,Lmatcol).*Hinttemp./obj.h*1e-9;
                                            intindex=intindex+1;
                                        else
                                            %every interachtion term is of the form
                                            %(L^-1)_ij*Phi_i*Phi_j with i!=j
                                            Hint(:,:,intindex)=Lmatg(Lmatrow,Lmatcol).*superkron(Hintern)./obj.h*1e-9;
                                            intindex=intindex+1;
                                        end
                                        %end
                                    end
                                end
                                if ~isempty(Cmatg) && ~isempty(find(Cmatg,1))
                                    for ind=find(triu(Cmatg,1))'
                                        [node1,node2]=ind2sub([length(Cmatg),length(Cmatg)],ind);
                                        circuitmarker=cumsum(sizes); %cumulative sum of the sequence of sizes of the capacitance
                                        %matrices
                                        circn1=find(node1<=circuitmarker,1);
                                        circn2=find(node2<=circuitmarker,1);
                                        if circn1>1
                                            node1=node1-(circuitmarker(circn1-1));
                                        end
                                        if circn2>1
                                            node2=node2-(circuitmarker(circn2-1));
                                        end
                                        if length(unique([sweep.qubitnum]))>1
                                                % toggle1=contains(unique([sweep.qubitnum]),circn1);
                                                % toggle2=contains(unique([sweep.qubitnum]),circn2);
                                            toggle1=~isempty(find([sweep.qubitnum]==circn1,1));
                                            toggle2=~isempty(find([sweep.qubitnum]==circn2,1));
                                        else
                                            toggle1=isequal(unique([sweep.qubitnum]),circn1);
                                            toggle2=isequal(unique([sweep.qubitnum]),circn2);
                                        end
                                        if iter==1 || toggle1
                                            v1=Evect{1,circn1}; %get the unperturbed eigenvectors
                                            Q1=Q{circn1}(:,:,node1);
                                            H1(intindex)={ncon({conj(v1),Q1,v1},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                            %H1(intindex)={ncon({v1,Q1,conj(v1)},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                        end
                                        if iter==1 || toggle2
                                            v2=Evect{1,circn2}; %get the unperturbed eigenvectors
                                            Q2=Q{circn2}(:,:,node2);
                                            H2(intindex)={ncon({conj(v2),Q2,v2},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                            %H2(intindex)={ncon({v2,Q2,conj(v2)},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                        end
                                        Hintern=Id;
                                        if circn1==circn2
                                            Hintern{1,circn1}=H1{intindex}*H2{intindex}; %mediated self-interaction term
                                        else
                                            Hintern{1,circn1}=H1{intindex};
                                            Hintern{1,circn2}=H2{intindex};
                                        end
                                        %                                             Hintern{1,circn1}=H1{intindex};
                                        %                                             Hintern{1,circn2}=H2{intindex};
                                        if ~isempty(obj.lev2truncation)
                                            Hinttemp=obj.lev2truncate(Hintern,U);
                                            Hint(:,:,intindex)=Cmatg(ind).*Hinttemp./obj.h*1e-9;
                                            intindex=intindex+1;
                                        else
                                            %every interachtion term is of the form
                                            %(C^-1)_ij*Q_i*Q_j with i!=j
                                            Hint(:,:,intindex)=Cmatg(ind)*superkron(Hintern)./obj.h*1e-9;
                                            intindex=intindex+1;
                                        end
                                    end
                                end
                                %sum unperturbed and interaction
                                %Hamiltonians
                                Htot=sum(Hbig,3)+sum(Hint,3);
                                opts.maxit=500;
                                eigenvals=eigs(sparse((Htot+Htot')./2),outlevels,'sr',opts);
                                out(iter,:)=sort(real(eigenvals));
%                                 out{1,end}(:,iter)=sort(real(eigenvals));
                                if nargin>3 && isequal(relative,'Relative') && optionr
                                    out(iter,:)=out(iter,:)-out(iter,1); %subtract ground state
%                                     out{1,end}(:,iter)=out{1,end}(:,iter)-out{1,end}(1,iter);
                                end
                            end
                            obj.resetloadparammats; %restore parammats
                        end
                        close(f)
                        if nargin>5 && isequal(plotopt,'Plot') && optionp
                            figure
                            plot(sweep(1).range,out,'LineWidth',2);
%                             plot(sweep(1).range,out{1,end},'LineWidth',2);
                            set(gca,'fontsize',18);
                            labelx=strcat('\textit{',sweep(1).parameter,'}${}_',num2str(sweep(1).qubitnum),'$');
                            xlim([min(sweep(1).range),max(sweep(1).range)]);
                            xlabel(labelx,'FontSize',22,'Interpreter','latex')
                            if nargin>3 && isequal(relative,'Relative') && optionr
                                labely=strcat('$E_i-E_0(',sweep(1).parameter,'{}_',num2str(sweep(1).qubitnum),')$ (GHz)');
                            else
                                labely=strcat('$E_i(',sweep(1).parameter,'{}_',num2str(sweep(1).qubitnum),')$ (GHz)');
                            end
                            ylabel(labely,'FontSize',22,'Interpreter','latex')
                        end
                    else
                        fprintf("Inconsistent params ranges\n");
                    end
                end
            end
        end
        
        %% Calculates the energy spectrum of the coupled system for a fixed parameter configuration
        % COEFFP  Determines the coefficient of the Pauli operators
        % opnames in the qubit Hamiltonian. The first input is a 1D or 2D 
        % cell array of strings of strings representing aliases for each 
        % Pauli operator we are interest in. The second input is a
        % numerical array containing the indices of every circuit to be
        % treated as a qubit (as opposed to a coupler). The result is
        % returned as a numerical array.
        function [out,Hpr]=coeffP(obj,opnames,qubitlist,method)
            numqubits=length(obj.circuitnames);
            if length(obj.interactions)<1
                fprintf("Define pairs of interacting qubits\n");
%             elseif length(obj.minductances)+length(obj.couplingcap)<length(obj.interpairs)
%                 fprintf("Define mutual inductances/shared capacitances values\n");
%             elseif length(obj.coupledinductors)+length(obj.couplednodes)<length(obj.interpairs)
%                 fprintf("Define coupling branch currents or node charges\n");
            elseif length(obj.lev1truncation)<numqubits
                fprintf("Define upper level truncation values\n");
            elseif length(opnames)~=numqubits && length(opnames{1})~=numqubits
                fprintf("Assign one operator to each circuit element\n");
            else
                if nargin<4
                    method=1;
                end
                obj.resetloadparammats;
                Ps=cell(1,numqubits);
                if size(opnames,1)>1
                    listlen=length(opnames{1});
                else
                    listlen=length(opnames);
                end
                if method==1
                    ops=cell(listlen,size(opnames,1)); %number of operators per
                    %list=listlen=numqubits, number of operators per each
                    %qubit=size(opnames,1)
                elseif method==2
                    ops=cell(numqubits,size(opnames,1));
                    rotations=cell(1,length(qubitlist));
                    firsten=zeros(1,length(qubitlist));
                end
                if size(opnames,1)>1
%                     for opind=1:listlen
%                         if any(find(qubitlist==opind)) %if it is a qubit
%                             ops(opind,:)=obj.circuits(opind).getPaulis(opnames(opind),2);
%                         end
%                     end
%                 else
                    opnm=cell(listlen,size(opnames,1));
                    for opind=1:listlen
                        opnm(opind,:)=mat2cell(reshape(cellfun(@(x) x{opind},opnames),[1 size(opnames,1)]),1,ones(1,size(opnames,1)));
%                         if any(find(qubitlist==opind)) %if it is a qubit
%                             ops(opind,:)=obj.circuits(opind).getPaulis(opn,2);
%                         end
                    end
                end
                if any(cellfun(@(x) isequal(x,'Ind'),{obj.interactions.ctype})) %If an inductive coupling is present
                    [dict,Lmatg]=obj.buildinductancemat(); %Get global branch inductance matrix
                    Lmatg=inv(Lmatg); %invert the inductance matrix
                else
                    Lmatg=[];
                end
                if any(cellfun(@(x) isequal(x,'Cap'),{obj.interactions.ctype})) %If a capacitive coupling is present
                    [~,Cmatg,sizes]=obj.buildcapacitancemat(); %Get inverse global node capacitance matrix
                else
                    Cmatg=[];
                end
                obj.setloadparammats; %Change capacitance and inductance matrices to account for capative/inductive loads
                numinter=length(find(triu(Lmatg,1)))+length(find(triu(Cmatg,1))); %Number of interaction terms
                %it equals half the number of off diagonal terms in the
                %inductance matrix plus half the number of block-off diagonal
                %terms in the capacitance matrix 
                if isempty(obj.lev2truncation) %if no second level truncation is defined
                    Hbig=zeros(prod(obj.lev1truncation)); %Sum of the single system hamiltonians
                    Hint=zeros(prod(obj.lev1truncation),prod(obj.lev1truncation),numinter); %Interaction hamiltonians
                else
                    if isempty(obj.lev3truncation)
                        dims=obj.lev2truncation;
                    else
                        dims=obj.lev3truncation;
                    end
                    subsnum=size(dims,1);
                    for subind=1:subsnum
                        if subind==1
                            sizetot=dims{subind,:}{2};
                        else
                            sizetot=sizetot*dims{subind,:}{2};
                        end
                    end
                    Hint=zeros(sizetot,sizetot,numinter);
                end
                Id=cellfun(@eye,num2cell(obj.lev1truncation),'UniformOutput',0);
                Evect=cell(1,numqubits); %Cell of eigenvectors
                H0lev2=cell(1,numqubits); %Unperturbed Hamiltonian for 2nd level hierarchical diagonalisation
                
                %Unperturbed Hamiltonians eigenvalues problem
                for nqubit=1:numqubits
                    Hsmall=obj.circuits(nqubit).getH;
                    %Hsmall=obj.circuits(nqubit).getH;
                    [Evect{1,nqubit},evals]=eigs(sparse(Hsmall),obj.lev1truncation(nqubit),'sr');
                    Htemp=Id;
                    Htemp(1,nqubit)={real(evals)};
                    if ~isempty(obj.lev2truncation)
                        H0lev2(1,nqubit)={real(evals)}; %copy single systems H0 for 2nd level h. diagonalisation
                    else
                        Hbig=Hbig+superkron(Htemp); %if no 2nd order h.d. is required, calculate the total H0
                        %i.e. the sum of the single circuits unperturbed
                        %Hamiltonians (corrected for capacitive/inductive
                        %loading)
                    end
                    if method==2
                        if any(find(qubitlist==nqubit))
                            firsten(qubitlist==nqubit)=real(evals(2,2));
                            if isequal(obj.circuits(nqubit).type,"flux")
                                Iop=obj.circuits(nqubit).getIops('Iz');
                            else
                                [~,Iop]=obj.circuits(nqubit).getPQ(false,true);
                                if length(size(Iop))==3
                                    Rinv=obj.circuits(nqubit).rotation';
                                    Iop=squeeze(ncon({Rinv(1,:),Iop},{[-1,1],[-2 -3 1]})); %Main qubit island must be on node 1!
%                                     Iop=sparse(Iop);
%                                 else
%                                     Iop=sparse(Iop);
                                end
                            end
                            Ip=ncon({conj(Evect{1,nqubit}(:,1:2)),Iop,Evect{1,nqubit}(:,1:2)},{[1 -1],[1 2],[2,-2]});
                            [ev,In]=eig(Ip);
                            [~,order]=sort(diag(In),'descend');
                            rotations{qubitlist==nqubit}=ev(:,order)';
                            for i=1:size(opnames,1)
                                if opnames{i}{nqubit}=="x"
                                    ops(nqubit,i)={[[0,1];[1,0]]};
                                elseif opnames{i}{nqubit}=="y"
                                    ops(nqubit,i)={[[0,-1i];[1i,0]]};
                                elseif opnames{i}{nqubit}=="z"
                                    ops(nqubit,i)={[[1,0];[0,-1]]};
                                elseif opnames{i}{nqubit}=="I"
                                    ops(nqubit,i)={eye(2)};
                                end
                            end
                        end
                    end
                end
                for oprow=1:listlen
                    ev=sparse(Evect{1,oprow});
                    if any(find(qubitlist==oprow)) %if it is a qubit
                        if isequal(obj.circuits(nqubit).type,"flux")
                            Iop=obj.circuits(oprow).getIops('Iz');
                        else
                            [~,Iop]=obj.circuits(oprow).getPQ(false,true);
                            if length(size(Iop))==3
                                Rinv=obj.circuits(oprow).rotation';
                                Iop=squeeze(ncon({Rinv(1,:),Iop},{[-1,1],[-2 -3 1]})); %Main qubit island must be on node 1!
                                %Iop=sparse(Iop);
%                             else
%                                 Iop=sparse(Iop);
                            end
                        end
                        %Iop=obj.circuits(oprow).getIops('Iz');
                        temp=zeros(size(ev,2));
                        temp(1,1)=1;
                        temp(2,2)=1;
                        Ps{oprow}=temp;
                        if size(opnames,1)==1
                            %optmpbis=obj.circuits(oprow).getPaulis(opnames{oprow},3,Iop,ev(:,1:2));
                            if method==1
                                ops(oprow,:)=obj.circuits(oprow).getPaulis(opnames{oprow},3,Iop,ev(:,1:2),size(ev,2));
                            end
                            %ops(oprow,:)={ncon({conj(ev),optmpbis{1},ev},{[1 -1],[1 2],[2,-2]})};
                        else
                            %optmpbis=obj.circuits(oprow).getPaulis(opnm(oprow,:),3,Iop,ev(:,1:2));
                            if method==1
                                ops(oprow,:)=obj.circuits(oprow).getPaulis(opnm(oprow,:),3,Iop,ev(:,1:2),size(ev,2));
                            end
                            %for index=1:length(optmpbis)
                            %    ops(oprow,index)={ncon({conj(ev),optmpbis{index},ev},{[1 -1],[1 2],[2,-2]})};
                            %end
                        end
%                         for opcol=1:size(opnames,1)
%                             ops{oprow,opcol}=ncon({conj(ev),ops{oprow,opcol},ev},{[1 -1],[1 2],[2,-2]});
%                         end
                    else
                        temp=zeros(size(ev,2));
                        temp(1,1)=1;
                        Ps{oprow}=temp;
                        for opcol=1:size(opnames,1)
                            if method==1
                                ops{oprow,opcol}=temp;
                            end
                        end
                    end
                end
                if ~isempty(obj.lev2truncation)
                    U=obj.lev2rotations(H0lev2); %calculate rotation mapping to the subsystems basis
                    Hbig=zeros(sizetot);
                    for nqubit=1:numqubits
                        Htemp=Id;
                        Htemp(1,nqubit)=H0lev2(1,nqubit);
                        Hbig=Hbig+obj.lev2truncate(Htemp,U); %apply rotation
%                         ops{nqubit}=obj.lev2truncate(ops{nqubit},U);
%                         Ps{nqubit}=obj.lev2truncate(Ps{nqubit},U);
                    end
                    opsout=cell(1,size(opnames,1));
                    for opind=1:size(opnames,1)
                        if method==1
                        %opstemp=mat2cell(reshape(cellfun(@(x) x{opind},ops),[1 size(opnames,1)]),1,ones(1,size(opnames,1)));
                            opsout{opind}=obj.lev2truncate(ops(:,opind)',U);
                        end
                    end
                    P0=obj.lev2truncate(Ps,U);
                else
                    P0=superkron(Ps);
                end
                intindex=1;
                if ~isempty(Lmatg)
    %                 Hint=zeros(prod(obj.lev1truncation),prod(obj.lev1truncation),intnum);
                    for Lind=find(triu(Lmatg,1))'%Lmatrow=1:length(Lmatg)
                        [Lmatrow,Lmatcol]=ind2sub(size(Lmatg),Lind);
                        %for Lmatcol=Lmatrow+1:length(Lmatg)
                        %for every element of the inductance matrix,
                        %find the involved circuits
                        circn1=dict(Lmatrow).circuit;
                        circn2=dict(Lmatcol).circuit;
                        v1=Evect{1,circn1}; %get the unperturbed eigenvectors
                        v2=Evect{1,circn2};
                        
                        %get the position of the two inductors to
                        %calculate the branch fluxes
                        inductn1=cellfun(@(x) isequal(x,dict(Lmatrow).inductor),{obj.circuits(circn1).inductors.names});
                        positions1=obj.circuits(dict(Lmatrow).circuit).inductors(inductn1).positions;
                        inductn2=cellfun(@(x) isequal(x,dict(Lmatcol).inductor),{obj.circuits(circn2).inductors.names});
                        positions2=obj.circuits(dict(Lmatcol).circuit).inductors(inductn2).positions;
                        [P1,~]=obj.circuits(circn1).getPQ;
                        [P2,~]=obj.circuits(circn2).getPQ;
                        Rinv1=inv(obj.circuits(circn1).rotation);
                        Rinv2=inv(obj.circuits(circn2).rotation);
                        if positions1(1)==positions1(2)
                            Pb1=squeeze(ncon({Rinv1(positions1(1),:),P1},{[-1,1],[-2 -3 1]}));
                        else
                            Pb1=squeeze(ncon({Rinv1(positions1(2),:),P1},{[-1,1],[-2 -3 1]}))-squeeze(ncon({Rinv1(positions1(1),:),P1},{[-1,1],[-2 -3 1]}));
                        end
                        if positions2(1)==positions2(2)
                            Pb2=squeeze(ncon({Rinv2(positions2(1),:),P2},{[-1,1],[-2 -3 1]}));
                        else
                            Pb2=squeeze(ncon({Rinv2(positions2(2),:),P2},{[-1,1],[-2 -3 1]}))-squeeze(ncon({Rinv2(positions2(1),:),P2},{[-1,1],[-2 -3 1]}));
                        end
                        %project the branch fluxes on the unperturbed
                        %Hamiltonian eigenvectors basis
                        H1=ncon({conj(v1),Pb1,v1},{[1 -1],[1 2],[2,-2]}); %Hint=H1*H2
                        H2=ncon({conj(v2),Pb2,v2},{[1 -1],[1 2],[2,-2]});
%                         H1=ncon({v1,Pb1,conj(v1)},{[1 -1],[1 2],[2,-2]}); %Hint=H1*H2
%                         H2=ncon({v2,Pb2,conj(v2)},{[1 -1],[1 2],[2,-2]});
%                         h1=H1(find(abs(H1)>1e-30,1));
%                         H1=H1./h1*abs(h1);
%                         h2=H2(find(abs(H2)>1e-30,1));
%                         H2=H2./h2*abs(h2);
                        Hintern=Id;
                        if circn1==circn2
                            Hintern{1,circn1}=H1*H2; %mediated self-interaction term
                        else
                            Hintern{1,circn1}=H1;
                            Hintern{1,circn2}=H2;
                        end
                        %every interachtion term is of the form
                            %(L^-1)_ij*Phi_i*Phi_j with i!=j
                        if ~isempty(obj.lev2truncation)
                            Hinttemp=obj.lev2truncate(Hintern,U); %map interaction terms to subsystem basis
                            Hint(:,:,intindex)=Lmatg(Lmatrow,Lmatcol).*Hinttemp./obj.h*1e-9;
                            intindex=intindex+1;
                            %fprintf(strcat('intindex=',num2str(intindex-1),', norm(H)=',num2str(norm(Hint(:,:,intindex-1)),'%e'),'\n'))
                        else
                            Hint(:,:,intindex)=Lmatg(Lmatrow,Lmatcol)*superkron(Hintern)./obj.h*1e-9;
                            intindex=intindex+1;
                            %fprintf(strcat('intindex=',num2str(intindex),', norm(H)=',num2str(norm(Hint(:,:,intindex)))))
                        end
                        %end
                    end
                end
                if ~isempty(Cmatg) && ~isempty(find(Cmatg,1))
                    for ind=find(triu(Cmatg,1))'
                        [node1,node2]=ind2sub([length(Cmatg),length(Cmatg)],ind);
                        circuitmarker=cumsum(sizes); %cumulative sum of the sequence of sizes of the capacitance
                        %matrices
                        circn1=find(node1<=circuitmarker,1);
                        circn2=find(node2<=circuitmarker,1);
                        v1=Evect{1,circn1}; %get the unperturbed eigenvectors
                        v2=Evect{1,circn2};
                        if circn1>1 %find the node number in the subcircuit
                            node1=node1-(circuitmarker(circn1-1));
                        end
                        if circn2>1
                            node2=node2-(circuitmarker(circn2-1));
                        end
                        [~,Q1]=obj.circuits(circn1).getPQ(false,true);
                        [~,Q2]=obj.circuits(circn2).getPQ(false,true);
                        Rinv1=obj.circuits(circn1).rotation';
                        Rinv2=obj.circuits(circn2).rotation';
                        Q1=squeeze(ncon({Rinv1(node1,:),Q1},{[-1,1],[-2 -3 1]}));
                        Q2=squeeze(ncon({Rinv2(node2,:),Q2},{[-1,1],[-2 -3 1]}));
                        %project the currents on the unperturbed
                        %Hamiltonian eigenvectors basis
                        H1=ncon({conj(v1),Q1,v1},{[1 -1],[1 2],[2,-2]}); %Hint=H1*H2
                        H2=ncon({conj(v2),Q2,v2},{[1 -1],[1 2],[2,-2]});
%                         H1=ncon({v1,Q1,conj(v1)},{[1 -1],[1 2],[2,-2]}); %Hint=H1*H2
%                         H2=ncon({v2,Q2,conj(v2)},{[1 -1],[1 2],[2,-2]});
                        Hintern=Id;
                        if circn1==circn2
                            Hintern{1,circn1}=H1*H2; %mediated self-interaction term
                        else
                            Hintern{1,circn1}=H1;
                            Hintern{1,circn2}=H2;
                        end
                        %every interaction term is of the form
                            %(C^-1)_ij*Q_i*Q_j with i!=j
                        if ~isempty(obj.lev2truncation) %map interaction terms to subsystem basis
                            Hinttemp=obj.lev2truncate(Hintern,U);
                            Hint(:,:,intindex)=Cmatg(ind).*Hinttemp./obj.h*1e-9;
                            intindex=intindex+1;
                        else
                            Hint(:,:,intindex)=Cmatg(ind)*superkron(Hintern)./obj.h*1e-9;
                            intindex=intindex+1;
                        end
                    end
                end
                Htot=Hbig+sum(Hint,3);
                %Htot=Hbig+sum(Hint,3);
                opts.maxit=500;
                %[eigvs,~]=eigs(sparse(Htot),max(20,2^length(qubitlist)),'sr',opts);
                [eigvs,en]=eigs(sparse(Htot),max(20,2^length(qubitlist)),'sr',opts);
                %eigvs=orth(eigvs(:,1:2^length(qubitlist))); %perform Gram-Schmidt orthogonalisation
                eigvs=eigvs(:,1:2^length(qubitlist));
                if method==1
                    P=ncon({eigvs,eye(2^length(qubitlist)),conj(eigvs)},{[-1 1],[1 2],[-2,2]});
                    U=sqrtm((2.*P0-eye(length(P0)))*(2.*P-eye(length(P0))));
                    %norm(P-P0)
                    Hpr=U*Htot*U';
                    out=zeros(1,size(opnames,1));
                    %a=cell(1,size(opnames,1));
                    for ind=1:size(opnames,1)
                        if ~isempty(obj.lev2truncation)
                            out(ind)=real(trace(opsout{1,ind}*Hpr))./(2^length(qubitlist));
                            %out(ind)=real(trace(opsout{1,ind}*Htot))./(2^length(qubitlist));
                        else
    %                         norm(imag(superkron(ops(:,ind))))
                            out(ind)=real(trace(superkron(ops(:,ind))*Hpr))./(2^length(qubitlist));
                            %out(ind)=real(trace(superkron(ops(:,ind))*Htot))./(2^length(qubitlist));
                            %a(ind)={superkron(ops(:,ind))};
                        end
                    end
                elseif method==2
                    %[eigvs2,~]=eigs(sparse(P0),2^length(qubitlist),'lr',opts);
                    eigvs2=zeros(length(Hbig),2^length(qubitlist)); %Needs to be generalised
                    zero1=zeros(obj.lev1truncation(1),1);
                    zero1(1)=1;
                    uno1=zeros(obj.lev1truncation(1),1);
                    uno1(2)=1;
                    zero2=zeros(obj.lev1truncation(2),1);
                    zero2(1)=1;
                    uno2=zeros(obj.lev1truncation(2),1);
                    uno2(2)=1;
                    eigvs2(:,1)=kron(zero1,zero2);
                    if firsten(2)<firsten(1)
                        eigvs2(:,2)=kron(zero1,uno2);
                        eigvs2(:,3)=kron(uno1,zero2);
                    else
                        eigvs2(:,2)=kron(uno1,zero2);
                        eigvs2(:,3)=kron(zero1,uno2);
                    end
                    eigvs2(:,4)=kron(uno1,uno2);
                    R=superkron(rotations);
%                     [eigvs0,~]=eigs(sparse(Hbig),max(20,2^length(qubitlist)),'sr',opts);
%                     eigvs0=eigvs0(:,1:2^length(qubitlist));
                    %[eigvs0,en0]=eigs(sparse(Hbig),max(20,2^length(qubitlist)),'sr',opts);
                    Up=ncon({eigvs,eigvs2},{[1 -1],[1,-2]});
                    %Up=ncon({eigvs0(:,1:2^length(qubitlist)),eigvs2},{[1 -1],[1,-2]});
                    Up=gsog(Up); %Gramm-Schmidt orthogonalisation
                    Up=bsxfun(@(x,y) x./y,Up,vecnorm(Up));
                    Hpr=en(1:2^length(qubitlist),1:2^length(qubitlist));
                    %Hpr=en0(1:2^length(qubitlist),1:2^length(qubitlist));
                    Hpr=R'*Up'*Hpr*Up*R;
                    out=zeros(1,size(opnames,1));
                    for ind=1:size(opnames,1)
                        out(ind)=real(trace(superkron(ops{:,ind})*Hpr))./(2^length(qubitlist));
                    end
                end
                obj.resetloadparammats; %reset capacitive and inductive matrices
            end
        end
        
        %% Calculates the energy spectrum of the coupled system as a function of one or more varying parameters
        % SWEEPCOEFFP Calculates the coefficient of the operators
        % opnames in the qubit Hamiltonian by projection, as a function
        % of a varying parameter. The first input is the sweep name
        % defined before, the second is a 1D or 2D cell array of strings
        % representing aliases for each Pauli operator we are interested
        % in. NB: a name needs to be provided for every circuit
        % (in the order of appearance in obj.circuitnames). "I" for
        % identity is used for coupler circuits. The third input is a
        % numerical array containing the indices of every circuit to be
        % treated as a qubit (as opposed to a coupler). Input 4 is the
        % 'Plot' option, which enables the final plot of the
        % results if input 5 is set to true. Input 6 defines a minimum
        % number of calculated levels used by the diagonalisation tools
        % (default is 2^N for N qubits). NB: the original circuit
        % properties/biases are restored at the end of the sweep.
        function out=sweepcoeffP(obj,sweepnm,opnames,qubitlist,plotopt,optionp,minlev,forcereal)
            numqubits=length(obj.circuitnames);
            if length(obj.interactions)<1
                fprintf("Define pairs of interacting qubits\n");
            elseif length(obj.lev1truncation)<numqubits
                fprintf("Define upper level truncation values\n");
            elseif nargin < 4
                fprintf("Specify sweepname, a set of Pauli operators and the list of qubit circuits\n");
            elseif isempty(obj.sweeps)
                fprintf("Define a sweep first using the function paramsweep or fluxsweep\n");
            elseif length(opnames)~=numqubits && length(opnames{1})~=numqubits
                fprintf("Assign one operator to each circuit element\n");
            else
                if nargin<7
                    minlev=20;
                end
                if nargin<8
                    forcereal=false;
                end
                obj.resetloadparammats;
                if any(cellfun(@(x) isequal(x,'Ind'),{obj.interactions.ctype}))
                    [dict,Lmatg]=obj.buildinductancemat(); %get global branch inductance matrix
                    Lmatg=inv(Lmatg); %invert the inductance matrix
                else
                    Lmatg=[];
                end
                if any(cellfun(@(x) isequal(x,'Cap'),{obj.interactions.ctype}))
                    [~,Cmatg,sizes]=obj.buildcapacitancemat(); %get inverse global node capacitance matrix
                else
                    Cmatg=[];
                end
                obj.setloadparammats;
                numinter=length(find(triu(Lmatg,1)))+length(find(triu(Cmatg,1))); %Number of interaction terms
                %it equals half the number of off diagonal terms in the
                %inductance matrix plus half the number of block-off diagonal
                %terms in the capacitance matrix 
                H1=cell(1,numinter);
                H2=cell(1,numinter);
                numsweptpar=0;
                toggle=1;
                %check that the specified sweep has been previously defined
                %(through paramsweep() or fluxsweep())
                for sweepn=1:length(obj.sweeps)
                    if isequal(obj.sweeps(sweepn).sweepname,sweepnm)
                        numsweptpar=numsweptpar+1; %if a sweep with the desired name is found, increase numsweptpar by 1
%                         if strcmp(obj.sweeps(sweepn).parameter(1),'L') || strcmp(obj.sweeps(sweepn).parameter(1),'C') || isequal(obj.sweeps(sweepn).parameter,'alpha') 
%                             %if the sweep changes the the flux/current operators, turn toggle on,
%                             toggle=1;
%                         end
                    end
                end
                if numsweptpar==0
                    fprintf("Specify an existent sweep name\n");
                else
                    %copy the sweep data
                    sweep=struct('sweepname',sweepnm,'qubitnum',cell(1,numsweptpar),'sweeptype',cell(1,numsweptpar),'parameter',...
                        cell(1,numsweptpar),'range',cell(1,numsweptpar));
                    param=1;
                    for sweepn=1:length(obj.sweeps)
                        if isequal(obj.sweeps(sweepn).sweepname,sweepnm)
                            sweep(param)=obj.sweeps(sweepn); % Only copy the sweeps with the right name
                            param=param+1;
                        end
                    end
                    sweeplengths=cellfun(@length,{sweep.range});
                    if all(sweeplengths==sweeplengths(1)) %checks that all swept parameters take the same number of values
                        out=zeros(size(opnames,1),length(sweep(1).range));
%                         out2=zeros(4,length(sweep(1).range));
                        if isequal(sweep(1).sweeptype,'param') %parameter sweep: requires calculating the full hamiltonian every time
                            paramold=zeros(1,numsweptpar); %copy the original value of the swept parameter
                            for paramn=1:numsweptpar
                                paramold(1,paramn)=obj.circuits(sweep(paramn).qubitnum).parameters.(sweep(paramn).parameter);
                            end
                            f=waitbar(1/length(sweep(1).range),strcat('Sweep progress:',num2str(100/length(sweep(1).range),2),'%'));
                            for iter=1:length(sweep(1).range) %for every point in the sweep
                                waitbar(iter/length(sweep(1).range),f,strcat('Sweep progress:',num2str(100*iter/length(sweep(1).range),2),'%'));
                                for paramn=1:numsweptpar %change qubit properties
                                    obj.circuits(sweep(paramn).qubitnum).parameters.(sweep(paramn).parameter)=sweep(paramn).range(iter);
                                end
                                if toggle %recalculate loaded matrices to account for a change in L or C
                                    obj.resetloadparammats;
                                    if any(cellfun(@(x) isequal(x,'Ind'),{obj.interactions.ctype}))
                                        [dict,Lmatg]=obj.buildinductancemat(); %get global branch inductance matrix
                                        Lmatg=inv(Lmatg); %invert the inductance matrix
                                    else
                                        Lmatg=[];
                                    end
                                    if any(cellfun(@(x) isequal(x,'Cap'),{obj.interactions.ctype}))
                                        [~,Cmatg,sizes]=obj.buildcapacitancemat(); %get inverse global node capacitance matrix
                                    else
                                        Cmatg=[];
                                    end
                                    obj.setloadparammats;
                                end
%                                 if iter==1
                                    %initialise cells at the first
                                    %iteration
                                    Ps=cell(1,numqubits);
                                    Iops=cell(1,numqubits);
                                    if size(opnames,1)>1
                                        listlen=length(opnames{1});
                                    else
                                        listlen=length(opnames);
                                    end
                                    ops=cell(listlen,size(opnames,1)); %number of operators per
                                    %list=listlen=numqubits, number of operators per each
                                    %qubit=size(opnames,1)
                                    %opscopy=cell(1,length(opnames));
                                    opscopy=cell(listlen,size(opnames,1));
                                    if size(opnames,1)>1
                                        opnm=cell(listlen,size(opnames,1));
                                        for opind=1:listlen
                                            opnm(opind,:)=mat2cell(reshape(cellfun(@(x) x{opind},opnames),[1 size(opnames,1)]),1,ones(1,size(opnames,1)));
                                        end
                                    end
                                    if isempty(obj.lev2truncation)
                                        Hbig=zeros(prod(obj.lev1truncation)); %Sum of the single system hamiltonians
                                        Hint=zeros(prod(obj.lev1truncation),prod(obj.lev1truncation),numinter); %Interaction hamiltonians
                                    else
                                        if isempty(obj.lev3truncation)
                                            dims=obj.lev2truncation;
                                        else
                                            dims=obj.lev3truncation;
                                        end
                                        subsnum=size(dims,1);
                                        for subind=1:subsnum
                                            if subind==1
                                                sizetot=dims{subind,:}{2};
                                            else
                                                sizetot=sizetot*dims{subind,:}{2};
                                            end
                                        end
                                        Hint=zeros(sizetot,sizetot,numinter);
                                    end
                                    Evect=cell(1,numqubits);
                                    H0lev2=cell(1,numqubits); %Unperturbed Hamiltonian for 2nd level hierarchical diagonalisation
                                    Id=cellfun(@eye,num2cell(obj.lev1truncation),'UniformOutput',0);
                                    P=cell(1,numqubits);
                                    Q=cell(1,numqubits);
                                    %calculate operators necessary for
                                    %interaction terms
                                    if ~isempty(Lmatg) %if there are iductive couplings
                                        for nqubit=1:numqubits %node fluxes
                                            [Ptemp,~]=obj.circuits(nqubit).getPQ;
                                            Rinv=inv(obj.circuits(nqubit).rotation);
                                            P(1,nqubit)={ncon({Rinv,Ptemp},{[-3,1],[-1 -2 1]})};
                                        end
                                    end
                                    if ~isempty(Cmatg) && ~isempty(find(Cmatg,1))
                                        for nqubit=1:numqubits %node charges
                                            [~,Qtemp]=obj.circuits(nqubit).getPQ;
                                            Rinv=obj.circuits(nqubit).rotation';
                                            Q(1,nqubit)={ncon({Rinv,Qtemp},{[-3,1],[-1 -2 1]})};
                                        end
                                    end
                                    %solve unperturbed Hamiltonian problems
                                    for nqubit=1:numqubits
                                        if forcereal==true
                                            Hsmall=real(obj.circuits(nqubit).getH);
                                        else
                                            Hsmall=obj.circuits(nqubit).getH;
                                        end
                                        if any(find(qubitlist==nqubit))
                                            Iops{nqubit}=obj.circuits(nqubit).getIops('Iz');
                                        end
%                                         Htemp=Id;
                                        [Evect{1,nqubit},evals]=eigs(sparse(Hsmall),obj.lev1truncation(nqubit),'SR');
%                                         [Evect{1,nqubit},Htemp{1,nqubit}]=eigs(Hsmall,obj.lev1truncation(nqubit),'SR');
%                                         [evals,order]=sort(real(diag(evals)));
%                                         Evect(1,nqubit)={Evecttemp(:,order)};
                                        Htemp=Id;
                                        Htemp(1,nqubit)={real(evals)};
                                        if ~isempty(obj.lev2truncation)
                                            H0lev2(1,nqubit)={real(evals)}; %copy single systems H0 for 2nd level h. diagonalisation
                                        else
                                            Hbig=Hbig+superkron(Htemp); %if no 2nd order h.d. is required, calculate the total H0
                                        end
                                    end
                                    for oprow=1:listlen
                                        ev=sparse(Evect{1,oprow});
                                        if any(find(qubitlist==oprow)) %if it is a qubit
                                            temp=zeros(size(ev,2));
                                            temp(1,1)=1;
                                            temp(2,2)=1;
                                            Ps{oprow}=temp;
                                            if size(opnames,1)==1
                                                opscopy(oprow)=obj.circuits(oprow).getPaulis(opnames{oprow},3,Iops{oprow},ev(:,1:2),size(ev,2));
                                            else
                                                opscopy(oprow,:)=obj.circuits(oprow).getPaulis(opnm(oprow,:),3,Iops{oprow},ev(:,1:2),size(ev,2));
                                            end
                                            %Evect{1,oprow}(:,1:2)=ev2;
                    %                         for opcol=1:size(opnames,1)
                    %                             ops{oprow,opcol}=ncon({conj(ev),ops{oprow,opcol},ev},{[1 -1],[1 2],[2,-2]});
                    %                         end
                                        else
                                            temp=zeros(size(ev,2));
                                            temp(1,1)=1;
                                            Ps{oprow}=temp;
                                            for opcol=1:size(opnames,1)
                                                opscopy{oprow,opcol}=temp;
                                            end
                                        end
                                    end
                                    if ~isempty(obj.lev2truncation)
                                        U=obj.lev2rotations(H0lev2);
                                        %                     Hbig=obj.lev2truncate(H0lev2,U);
                                        Hbig=zeros(sizetot);
                                        for nqubit=1:numqubits
                                            Htemp=Id;
                                            Htemp(1,nqubit)=H0lev2(1,nqubit);
                                            Hbig=Hbig+obj.lev2truncate(Htemp,U);
                                            size(Hbig)
%                                             opscopy{nqubit}=obj.lev2truncate(opscopy{nqubit},U);
%                                             Ps{nqubit}=obj.lev2truncate(Ps{nqubit},U);
                                        end
                                        opsout=cell(1,size(opnames,1));
                                        for opind=1:size(opnames,1)
                                            %opstemp=mat2cell(reshape(cellfun(@(x) x{opind},ops),[1 size(opnames,1)]),1,ones(1,size(opnames,1)));
                                            opsout{opind}=obj.lev2truncate(opscopy(:,opind)',U);
                                        end
                                        P0=obj.lev2truncate(Ps,U);
                                    else
                                        P0=superkron(Ps);
                                    end
                                    %find the interaction terms
                                    intindex=1;
                                    if ~isempty(Lmatg)
                                        for Lind=find(triu(Lmatg,1))'%Lmatrow=1:length(Lmatg)
                                            [Lmatrow,Lmatcol]=ind2sub(size(Lmatg),Lind);%for Lmatcol=Lmatrow+1:length(Lmatg)
                                            %for every element of the inductance matrix,
                                            %find the involved circuits
                                            circn1=dict(Lmatrow).circuit;
                                            circn2=dict(Lmatcol).circuit;

                                            v1=Evect{1,circn1}; %get the unperturbed eigenvectors
                                            inductn1=cellfun(@(x) isequal(x,dict(Lmatrow).inductor),{obj.circuits(circn1).inductors.names});
                                            positions1=obj.circuits(dict(Lmatrow).circuit).inductors(inductn1).positions;
                                            if positions1(1)==positions1(2)
                                                Pb1=P{circn1}(:,:,positions1(1));
                                            else
                                                Pb1=P{circn1}(:,:,positions1(2))-P{circn1}(:,:,positions1(1));
                                            end
                                            H1(intindex)={ncon({conj(v1),Pb1,v1},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                            %H1(intindex)={ncon({v1,Pb1,conj(v1)},{[1 -1],[1 2],[2,-2]})};

                                            v2=Evect{1,circn2}; %get the unperturbed eigenvectors
                                            inductn2=cellfun(@(x) isequal(x,dict(Lmatcol).inductor),{obj.circuits(circn2).inductors.names});
                                            positions2=obj.circuits(dict(Lmatcol).circuit).inductors(inductn2).positions;
                                            if positions2(1)==positions2(2)
                                                Pb2=P{circn2}(:,:,positions2(1));
                                            else
                                                Pb2=P{circn2}(:,:,positions2(2))-P{circn2}(:,:,positions2(1));
                                            end
                                            H2(intindex)={ncon({conj(v2),Pb2,v2},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                            %H2(intindex)={ncon({v2,Pb2,conj(v2)},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2

                                            Hintern=Id;
                                            if circn1==circn2
                                                Hintern{1,circn1}=H1{intindex}*H2{intindex}; %mediated self-interaction term
                                            else
                                                Hintern{1,circn1}=H1{intindex};
                                                Hintern{1,circn2}=H2{intindex};
                                            end
                                            if ~isempty(obj.lev2truncation)
                                                Hinttemp=obj.lev2truncate(Hintern,U);
                                                Hint(:,:,intindex)=Lmatg(Lmatrow,Lmatcol).*Hinttemp./obj.h*1e-9;
                                                intindex=intindex+1;
                                            else
                                                %every interachtion term is of the form
                                                %(L^-1)_ij*Phi_i*Phi_j with i!=j
                                                Hint(:,:,intindex)=Lmatg(Lmatrow,Lmatcol)*superkron(Hintern)./obj.h*1e-9;
                                                intindex=intindex+1;
                                            end
                                            %end
                                        end
                                    end
                                    if ~isempty(Cmatg) && ~isempty(find(Cmatg,1))
                                        for ind=find(triu(Cmatg,1))'
                                            %for every element of the inverse matrix,
                                            %find the involved circuits
                                            [node1,node2]=ind2sub([length(Cmatg),length(Cmatg)],ind);
                                            circuitmarker=cumsum(sizes); %cumulative sum of the sequence of sizes of the capacitance
                                            %matrices
                                            circn1=find(node1<=circuitmarker,1);
                                            circn2=find(node2<=circuitmarker,1);
                                            if circn1>1 %find the node index in the subsystem
                                                node1=node1-(circuitmarker(circn1-1));
                                            end
                                            
                                            v1=Evect{1,circn1}; %get the unperturbed eigenvectors
                                            Q1=Q{circn1}(:,:,node1);
                                            H1(intindex)={ncon({conj(v1),Q1,v1},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                            %H1(intindex)={ncon({v1,Q1,conj(v1)},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2

                                            v2=Evect{1,circn2}; %get the unperturbed eigenvectors
                                            Q2=Q{circn2}(:,:,node2);
                                            H2(intindex)={ncon({conj(v2),Q2,v2},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                            %H2(intindex)={ncon({v2,Q2,conj(v2)},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                            
                                            Hintern=Id;
                                            if circn1==circn2
                                                Hintern{1,circn1}=H1{intindex}*H2{intindex}; %mediated self-interaction term
                                            else
                                                Hintern{1,circn1}=H1{intindex};
                                                Hintern{1,circn2}=H2{intindex};
                                            end
                                            if ~isempty(obj.lev2truncation)
                                                Hinttemp=obj.lev2truncate(Hintern,U);
                                                Hint(:,:,intindex)=Cmatg(ind).*Hinttemp./obj.h*1e-9;
                                                intindex=intindex+1;
                                            else
                                                %every interachtion term is of the form
                                                %(C^-1)_ij*Q_i*Q_j with i!=j
                                                Hint(:,:,intindex)=Cmatg(ind)*superkron(Hintern)./obj.h*1e-9;
                                                intindex=intindex+1;
                                            end
                                        end
                                    end
                                %sum unperturbed and perturbed Hamiltonians
                                Htot=sum(Hbig,3)+sum(Hint,3);
                                opts.maxit=500;
                                [eigvs,~]=eigs(sparse((Htot+Htot')./2),max(minlev,2^length(qubitlist)),'sr',opts);
                                eigvs=eigvs(:,1:2^length(qubitlist));
                                Pj=ncon({eigvs,eye(2^length(qubitlist)),conj(eigvs)},{[-1 1],[1 2],[-2,2]});
                                U=sqrtm((2.*P0-eye(length(P0)))*(2.*Pj-eye(length(P0))));
                                Hpr=U*(Htot+Htot')*U'./2;
                                for ind=1:size(opnames,1)
                                    if ~isempty(obj.lev2truncation)
                                        out(ind,iter)=real(trace(opsout{1,ind}*Hpr)./(2^length(qubitlist)));
                                    else
                                        out(ind,iter)=real(trace(superkron(opscopy(:,ind)')*Hpr)./(2^length(qubitlist)));
                                    end
                                end
                            end
                            obj.resetloadparammats; %restore parammats
                            for paramn=1:numsweptpar %restore original param values
                                obj.circuits(sweep(paramn).qubitnum).parameters.(sweep(paramn).parameter)=paramold(1,paramn);
                            end
                        elseif isequal(sweep(1).sweeptype,'flux') %flux sweep, only the bias matrix needs to be recalculated
                            f=waitbar(1/length(sweep(1).range),strcat('Sweep progress:',num2str(100/length(sweep(1).range),2),'%'));
                            for iter=1:length(sweep(1).range) %for every point in the sweep
                                waitbar(iter/length(sweep(1).range),f,strcat('Sweep progress:',num2str(100*iter/length(sweep(1).range),2),'%'));
                                if iter==1
                                    Ps=cell(1,numqubits);
                                    if size(opnames,1)>1
                                        listlen=length(opnames{1});
                                    else
                                        listlen=length(opnames);
                                    end
                                    ops=cell(listlen,size(opnames,1)); %number of operators per
                                    %list=listlen=numqubits, number of operators per each
                                    %qubit=size(opnames,1)
                                    %opscopy=cell(1,length(opnames));
                                    opscopy=cell(listlen,size(opnames,1));
                                    if size(opnames,1)>1
                                        opnm=cell(listlen,size(opnames,1));
                                        for opind=1:listlen
                                            opnm(opind,:)=mat2cell(reshape(cellfun(@(x) x{opind},opnames),[1 size(opnames,1)]),1,ones(1,size(opnames,1)));
                    %                         if any(find(qubitlist==opind)) %if it is a qubit
                    %                             ops(opind,:)=obj.circuits(opind).getPaulis(opn,2);
                    %                         end
                                        end
                                    end
                                    %initialise cells at the first
                                    %iteration
                                    Id=cellfun(@eye,num2cell(obj.lev1truncation),'UniformOutput',0);
                                    if isempty(obj.lev2truncation)
%                                         Hbig=zeros(prod(obj.lev1truncation)); %Sum of the single system hamiltonians
                                        Hbig=zeros(prod(obj.lev1truncation),prod(obj.lev1truncation),numqubits); %Single system hamiltonians
                                        Hint=zeros(prod(obj.lev1truncation),prod(obj.lev1truncation),numinter); %Interaction hamiltonians
                                    else
                                        if isempty(obj.lev3truncation)
                                            dims=obj.lev2truncation;
                                        else
                                            dims=obj.lev3truncation;
                                        end
                                        subsnum=size(dims,1);
                                        for subind=1:subsnum
                                            if subind==1
                                                sizetot=dims{subind,:}{2};
                                            else
                                                sizetot=sizetot*dims{subind,:}{2};
                                            end
                                        end
                                        Hint=zeros(sizetot,sizetot,numinter);
                                    end
                                    HLC=cell(1,numqubits);
                                    Dmat=cell(1,numqubits);
                                    Evect=cell(1,numqubits);
                                    H0lev2=cell(1,numqubits); %Unperturbed Hamiltonian for 2nd level hierarchical diagonalisation
%                                     Hint=zeros(prod(obj.lev1truncation),prod(obj.lev1truncation),numinter); %interaction hamiltonians
                                    P=cell(1,numqubits);
                                    Q=cell(1,numqubits);
                                    %calculate operators necessary for
                                    %interaction terms
                                    if ~isempty(Lmatg) %if there are iductive couplings
                                        for nqubit=1:numqubits %node fluxes
                                            [Ptemp,~]=obj.circuits(nqubit).getPQ;
                                            Rinv=inv(obj.circuits(nqubit).rotation);
                                            P(1,nqubit)={ncon({Rinv,Ptemp},{[-3,1],[-1 -2 1]})};
                                        end
                                    end
                                    if ~isempty(Cmatg) && ~isempty(find(Cmatg,1))
                                        for nqubit=1:numqubits %node charges
                                            [~,Qtemp]=obj.circuits(nqubit).getPQ;
                                            Rinv=obj.circuits(nqubit).rotation';
                                            Q(1,nqubit)={ncon({Rinv,Qtemp},{[-3,1],[-1 -2 1]})};
                                        end
                                    end
                                    nummodes=zeros(1,numqubits);
                                    Iops=cell(1,numqubits);
                                    for nqubit=1:numqubits
                                        if any(find(qubitlist==nqubit))
                                            Iops{nqubit}=obj.circuits(nqubit).getIops('Iz');
                                        end
                                        %unperturbed Hamiltonians
                                        nummodes(nqubit)=length(obj.circuits(nqubit).modes);
                                        HLC(1,nqubit)={obj.circuits(nqubit).getHLC};
                                        Dmat(1,nqubit)={obj.circuits(nqubit).getDops};
                                        EJ=obj.circuits(nqubit).parammat.Ejmat;
                                        HJ=sum(sum(EJ)).*eye(length(HLC{1,nqubit}));
                                        if any(find([sweep.qubitnum]==nqubit))
                                            biasmatrix=obj.circuits(nqubit).getBiasm(sweep([sweep.qubitnum]==nqubit),1); % pass the value from the sweep only to the qubits
                                        else                                                      % affected by the sweep
                                            biasmatrix=obj.circuits(nqubit).getBiasm();
                                        end
                                        for ind=find(EJ)'
                                            [Dmatrow,Dmatcolumn]=ind2sub(size(EJ,1),ind);
                                            HJ=HJ-EJ(Dmatrow,Dmatcolumn)/2.*(biasmatrix(Dmatrow,Dmatcolumn).*Dmat{1,nqubit}{Dmatrow,Dmatcolumn}+...
                                                conj(biasmatrix(Dmatrow,Dmatcolumn)).*Dmat{1,nqubit}{Dmatrow,Dmatcolumn}');
%                                                 end
%                                             end
                                        end
                                        HJ=HJ./obj.h*1e-9;
                                        if forcereal==true
                                            [Evect{1,nqubit},evals]=eigs(sparse(real(HLC{1,nqubit}+HJ)),obj.lev1truncation(nqubit),'SR');
                                        else
                                            [Evect{1,nqubit},evals]=eigs(sparse(HLC{1,nqubit}+HJ),obj.lev1truncation(nqubit),'SR');
                                        end
                                        Htemp=Id;
                                        Htemp(1,nqubit)={real(evals)};
                                        if ~isempty(obj.lev2truncation)
                                            H0lev2(1,nqubit)={real(evals)}; %copy single systems H0 for 2nd level h. diagonalisation
                                        else
                                            Hbig(:,:,nqubit)=superkron(Htemp); %if no 2nd order h.d. is required, calculate the total H0
                                        end
                                    end
                                    for oprow=1:listlen
                                        ev=sparse(Evect{1,oprow});
                                        if any(find(qubitlist==oprow)) %if it is a qubit
                                            temp=zeros(size(ev,2));
                                            temp(1,1)=1;
                                            temp(2,2)=1;
                                            Ps{oprow}=temp;
                                            if size(opnames,1)==1
                                                opscopy(oprow)=obj.circuits(oprow).getPaulis(opnames{oprow},3,Iops{oprow},ev(:,1:2),size(ev,2));
                                            else
                                                opscopy(oprow,:)=obj.circuits(oprow).getPaulis(opnm(oprow,:),3,Iops{oprow},ev(:,1:2),size(ev,2));
                                            end
                                        else
                                            temp=zeros(size(ev,2));
                                            temp(1,1)=1;
                                            Ps{oprow}=temp;
                                            for opcol=1:size(opnames,1)
                                                opscopy{oprow,opcol}=temp;
                                            end
                                        end
                                    end
                                    if ~isempty(obj.lev2truncation)
                                        U=obj.lev2rotations(H0lev2);
                                        %                     Hbig=obj.lev2truncate(H0lev2,U);
                                        Hbig=zeros(sizetot);
                                        for nqubit=1:numqubits
                                            Htemp=Id;
                                            Htemp(1,nqubit)=H0lev2(1,nqubit);
                                            Hbig=Hbig+obj.lev2truncate(Htemp,U);
                                            %opscopy{nqubit}=obj.lev2truncate(opscopy{nqubit},U);
                                            %Ps{nqubit}=obj.lev2truncate(Ps{nqubit},U);
                                        end
                                        opsout=cell(1,size(opnames,1));
                                        for opind=1:size(opnames,1)
                                            %opstemp=mat2cell(reshape(cellfun(@(x) x{opind},ops),[1 size(opnames,1)]),1,ones(1,size(opnames,1)));
                                            opsout{opind}=obj.lev2truncate(opscopy(:,opind)',U);
                                        end
                                        P0=obj.lev2truncate(Ps,U);
                                    else
                                        P0=superkron(Ps);
                                    end
                                    
                                else
                                    for nqubit=unique([sweep.qubitnum]) %update only the unperturbed Hamiltonians
                                        %relative to qubits involved in the sweep
                                        EJ=obj.circuits(nqubit).parammat.Ejmat;
                                        HJ=sum(sum(EJ)).*eye(length(HLC{1,nqubit}));
                                        %update the biases values in the
                                        %Josephson terms
                                        biasmatrix=obj.circuits(nqubit).getBiasm(sweep([sweep.qubitnum]==nqubit),iter);
                                        for ind=find(EJ)'
                                            [Dmatrow,Dmatcolumn]=ind2sub(size(EJ,1),ind);
%                                         for Dmatrow=1:nummodes(nqubit)
%                                             for Dmatcolumn=Dmatrow:nummodes(nqubit)
%                                                 if EJ(Dmatrow,Dmatcolumn)~=0 %sum only on the branches containing a JJ
                                            HJ=HJ-EJ(Dmatrow,Dmatcolumn)/2.*(biasmatrix(Dmatrow,Dmatcolumn).*Dmat{1,nqubit}{Dmatrow,Dmatcolumn}+...
                                                conj(biasmatrix(Dmatrow,Dmatcolumn)).*Dmat{1,nqubit}{Dmatrow,Dmatcolumn}');
%                                                 end
%                                             end
                                        end
                                        HJ=HJ./obj.h*1e-9;
%                                         Htemp=Id;
                                        if forcereal==true
                                            [Evect{1,nqubit},evals]=eigs(sparse(real(HLC{1,nqubit}+HJ)),obj.lev1truncation(nqubit),'SR');
                                        else
                                            [Evect{1,nqubit},evals]=eigs(sparse(HLC{1,nqubit}+HJ),obj.lev1truncation(nqubit),'SR');
                                        end
                                        Htemp=Id;
                                        Htemp(1,nqubit)={real(evals)};
                                        if any(find(qubitlist==nqubit)) %if it is a qubit
                                            evtmp=Evect{1,nqubit};
                                            %opscopy{nqubit}=ncon({conj(Evect{1,nqubit}),ops{nqubit},Evect{1,nqubit}},{[1 -1],[1 2],[2,-2]});
                                            if size(opnames,1)==1
                                                opscopy(nqubit)=obj.circuits(nqubit).getPaulis(opnames{nqubit},3,Iops{nqubit},evtmp(:,1:2),size(evtmp,2));
                                            else
                                                opscopy(nqubit,:)=obj.circuits(nqubit).getPaulis(opnm(nqubit,:),3,Iops{nqubit},evtmp(:,1:2),size(evtmp,2));
                                            end
                                        end
                                        if ~isempty(obj.lev2truncation)
                                            H0lev2(1,nqubit)={real(evals)}; %copy single systems H0 for 2nd level h. diagonalisation
                                        else
                                            Hbig(:,:,nqubit)=superkron(Htemp); %if no 2nd order h.d. is required, calculate the total H0
                                        end
                                    end
                                    if ~isempty(obj.lev2truncation)
                                        U=obj.lev2rotations(H0lev2);
                                        %                     Hbig=obj.lev2truncate(H0lev2,U);
                                        Hbig=zeros(sizetot);
                                        for nqubit=1:numqubits
                                            Htemp=Id;
                                            Htemp(1,nqubit)=H0lev2(1,nqubit);
                                            Hbig=Hbig+obj.lev2truncate(Htemp,U);
                                        end
                                        opsout=cell(1,size(opnames,1));
                                        for opind=1:size(opnames,1)
                                            %opstemp=mat2cell(reshape(cellfun(@(x) x{opind},ops),[1 size(opnames,1)]),1,ones(1,size(opnames,1)));
                                            opsout{opind}=obj.lev2truncate(opscopy(:,opind)',U);
                                        end
                                    end
                                end
                                intindex=1;
                                if ~isempty(Lmatg)
                                    for Lind=find(triu(Lmatg,1))'%Lmatrow=1:length(Lmatg)
                                        [Lmatrow,Lmatcol]=ind2sub(size(Lmatg),Lind);%for Lmatcol=Lmatrow+1:length(Lmatg)
                                        %for every element of the inductance matrix,
                                        %find the involved circuits
                                        circn1=dict(Lmatrow).circuit;
                                        circn2=dict(Lmatcol).circuit;
                                        if length(unique([sweep.qubitnum]))>1 
                                            toggle1=~isempty(find([sweep.qubitnum]==circn1, 1));
                                            toggle2=~isempty(find([sweep.qubitnum]==circn2, 1));
                                        else
                                            toggle1=isequal(unique([sweep.qubitnum]),circn1);
                                            toggle2=isequal(unique([sweep.qubitnum]),circn2);
                                        end
                                        if iter==1 || toggle1 %if so, recalculate expectation values
                                            %on the new eigenstates
                                            v1=Evect{1,circn1}; %get the unperturbed eigenvectors
                                            inductn1=cellfun(@(x) isequal(x,dict(Lmatrow).inductor),{obj.circuits(circn1).inductors.names});
                                            positions1=obj.circuits(dict(Lmatrow).circuit).inductors(inductn1).positions;
                                            if positions1(1)==positions1(2)
                                                Pb1=P{circn1}(:,:,positions1(1));
                                            else
                                                Pb1=P{circn1}(:,:,positions1(2))-P{circn1}(:,:,positions1(1));
                                            end
                                            H1(intindex)={ncon({conj(v1),Pb1,v1},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                        end
                                        if iter==1 || toggle2
                                            v2=Evect{1,circn2}; %get the unperturbed eigenvectors
                                            inductn2=cellfun(@(x) isequal(x,dict(Lmatcol).inductor),{obj.circuits(circn2).inductors.names});
                                            positions2=obj.circuits(dict(Lmatcol).circuit).inductors(inductn2).positions;
                                            if positions2(1)==positions2(2)
                                                Pb2=P{circn2}(:,:,positions2(1));
                                            else
                                                Pb2=P{circn2}(:,:,positions2(2))-P{circn2}(:,:,positions2(1));
                                            end
                                            H2(intindex)={ncon({conj(v2),Pb2,v2},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                        end
                                        Hintern=Id;
                                        if circn1==circn2
                                            Hintern{1,circn1}=H1{intindex}*H2{intindex}; %mediated self-interaction term
                                        else
                                            Hintern{1,circn1}=H1{intindex};
                                            Hintern{1,circn2}=H2{intindex};
                                        end
                                        if ~isempty(obj.lev2truncation)
                                            Hinttemp=obj.lev2truncate(Hintern,U);
                                            Hint(:,:,intindex)=Lmatg(Lmatrow,Lmatcol).*Hinttemp./obj.h*1e-9;
                                            intindex=intindex+1;
                                        else
                                            %every interachtion term is of the form
                                            %(L^-1)_ij*Phi_i*Phi_j with i!=j
                                            Hint(:,:,intindex)=Lmatg(Lmatrow,Lmatcol).*superkron(Hintern)./obj.h*1e-9;
                                            intindex=intindex+1;
                                        end
                                        %end
                                    end
                                end
                                if ~isempty(Cmatg) && ~isempty(find(Cmatg,1))
                                    for ind=find(triu(Cmatg,1))'
                                        [node1,node2]=ind2sub([length(Cmatg),length(Cmatg)],ind);
                                        circuitmarker=cumsum(sizes); %cumulative sum of the sequence of sizes of the capacitance
                                        %matrices
                                        circn1=find(node1<=circuitmarker,1);
                                        circn2=find(node2<=circuitmarker,1);
                                        if circn1>1
                                            node1=node1-(circuitmarker(circn1-1));
                                        end
                                        if circn2>1
                                            node2=node2-(circuitmarker(circn2-1));
                                        end
                                        if length(unique([sweep.qubitnum]))>1
                                            toggle1=~isempty(find([sweep.qubitnum]==circn1,1));
                                            toggle2=~isempty(find([sweep.qubitnum]==circn2,1));
                                        else
                                            toggle1=isequal(unique([sweep.qubitnum]),circn1);
                                            toggle2=isequal(unique([sweep.qubitnum]),circn2);
                                        end
                                        if iter==1 || toggle1
                                            v1=Evect{1,circn1}; %get the unperturbed eigenvectors
                                            Q1=Q{circn1}(:,:,node1);
                                            H1(intindex)={ncon({conj(v1),Q1,v1},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                        end
                                        if iter==1 || toggle2
                                            v2=Evect{1,circn2}; %get the unperturbed eigenvectors
                                            Q2=Q{circn2}(:,:,node2);
                                            H2(intindex)={ncon({conj(v2),Q2,v2},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                        end
                                        Hintern=Id;
                                        if circn1==circn2
                                            Hintern{1,circn1}=H1{intindex}*H2{intindex}; %mediated self-interaction term
                                        else
                                            Hintern{1,circn1}=H1{intindex};
                                            Hintern{1,circn2}=H2{intindex};
                                        end
                                        %                                             Hintern{1,circn1}=H1{intindex};
                                        %                                             Hintern{1,circn2}=H2{intindex};
                                        if ~isempty(obj.lev2truncation)
                                            Hinttemp=obj.lev2truncate(Hintern,U);
                                            Hint(:,:,intindex)=Cmatg(ind).*Hinttemp./obj.h*1e-9;
                                            intindex=intindex+1;
                                        else
                                            %every interachtion term is of the form
                                            %(C^-1)_ij*Q_i*Q_j with i!=j
                                            Hint(:,:,intindex)=Cmatg(ind)*superkron(Hintern)./obj.h*1e-9;
                                            intindex=intindex+1;
                                        end
                                    end
                                end
                                %sum unperturbed and interaction
                                %Hamiltonians
                                H0=sum(Hbig,3);
                                if forcereal==true
                                    Htot=real(H0+sum(Hint,3));
                                else
                                    Htot=H0+sum(Hint,3);
                                end
                                opts.maxit=500;
                                [eigvs,~]=eigs(sparse((Htot+Htot')./2),max(minlev,2^length(qubitlist)),'sr',opts);
                                eigvs=eigvs(:,1:2^length(qubitlist));
                                Pj=ncon({eigvs,eye(2^length(qubitlist)),conj(eigvs)},{[-1 1],[1 2],[-2,2]});
                                U=sqrtm((2.*P0-eye(length(P0)))*(2.*Pj-eye(length(P0))));
                                Hpr=U*(Htot+Htot')*U'./2;
                                for ind=1:size(opnames,1)
                                    if ~isempty(obj.lev2truncation)
                                        out(ind,iter)=real(trace(opsout{1,ind}*(Hpr+Hpr')./2))./(2^length(qubitlist));
                                    else
                                        out(ind,iter)=real(trace(superkron(opscopy(:,ind)')*(Hpr+Hpr')./2))./(2^length(qubitlist));
                                    end
                                end
                            end
                            obj.resetloadparammats; %restore parammats
                        end
                        close(f)
                        if nargin>5 && isequal(plotopt,'Plot') && optionp
                            figure
                            plot(sweep(1).range,out,'LineWidth',2);
                            set(gca,'fontsize',18);
                            labelx=strcat('\textit{',sweep(1).parameter,'}${}_',num2str(sweep(1).qubitnum),'$');
                            xlim([min(sweep(1).range),max(sweep(1).range)]);
                            xlabel(labelx,'FontSize',22,'Interpreter','latex')
                            if size(opnames,1)==1
                                opnames=opnames(qubitlist);
                                opnames=cellfun(@(x) strcat('\sigma_',x),opnames);
                                labely=strcat('$',strjoin(cellstr(opnames),''),'$ coefficient $(',sweep(1).parameter,'{}_',num2str(sweep(1).qubitnum),')$ (GHz)');
                                ylabel(labely,'FontSize',22,'Interpreter','latex')
                            else
                                leglabel=cell(1,size(opnames,1));
                                for ind=1:size(opnames,1)
                                    opnamescopy=opnames{ind}(qubitlist);
                                    leglabel{ind}=strcat('$',strjoin(cellfun(@(x) strcat('\sigma_',x),opnamescopy,'UniformOutput',false),''),'$');
                                end
                                legend(leglabel,'FontSize',22,'Interpreter','latex')
                                legend('boxoff')
                                ylabel('Coefficients (GHz)','FontSize',22,'Interpreter','latex')
                            end
                        end
                    else
                        fprintf("Inconsistent params ranges\n");
                    end
                end
            end
        end
        
        %% Calculates the energy spectrum of the coupled system for a fixed parameter configuration
        % MATELEM Calculates the matrix element of a current/charge
        % operator.
        %
        %   out=obj.matelem(circuitnum,'iopname',initstate,finstate) Calculates the
        %   matrix element of the operator 'iopname' of circuit number 
        %   circuitnum between the energy eigenstates initstate and
        %   finstate.
        function out=matelem(obj,circuitnum,iopname,initstate,finstate)
            numqubits=length(obj.circuitnames);
            if length(obj.interactions)<1
                fprintf("Define pairs of interacting qubits\n");
            elseif length(obj.lev1truncation)<numqubits
                fprintf("Define upper level truncation values\n");
            else
                if any(cellfun(@(x) isequal(x,'Ind'),{obj.interactions.ctype})) %If an inductive coupling is present
                    [dict,Lmatg]=obj.buildinductancemat(); %Get global branch inductance matrix
                    Lmatg=inv(Lmatg); %invert the inductance matrix
                else
                    Lmatg=[];
                end
                if any(cellfun(@(x) isequal(x,'Cap'),{obj.interactions.ctype})) %If a capacitive coupling is present
                    [~,Cmatg,sizes]=obj.buildcapacitancemat(); %Get inverse global node capacitance matrix
                else
                    Cmatg=[];
                end
                obj.setloadparammats; %Change capacitance and inductance matrices to account for capative/inductive loads
                numinter=length(find(triu(Lmatg,1)))+length(find(triu(Cmatg,1))); %Number of interaction terms
                %it equals half the number of off diagonal terms in the
                %inductance matrix plus half the number of block-off diagonal
                %terms in the capacitance matrix 
                if isempty(obj.lev2truncation) %if no second level truncation is defined
                    Hbig=zeros(prod(obj.lev1truncation)); %Sum of the single system hamiltonians
                    Hint=zeros(prod(obj.lev1truncation),prod(obj.lev1truncation),numinter); %Interaction hamiltonians
                else
                    if isempty(obj.lev3truncation)
                        dims=obj.lev2truncation;
                    else
                        dims=obj.lev3truncation;
                    end
                    subsnum=size(dims,1);
                    for subind=1:subsnum
                        if subind==1
                            sizetot=dims{subind,:}{2};
                        else
                            sizetot=sizetot*dims{subind,:}{2};
                        end
                    end
                    Hint=zeros(sizetot,sizetot,numinter);
                end
                Id=cellfun(@eye,num2cell(obj.lev1truncation),'UniformOutput',0);
                Evect=cell(1,numqubits); %Cell of eigenvectors
                H0lev2=cell(1,numqubits); %Unperturbed Hamiltonian for 2nd level hierarchical diagonalisation
                
                %Unperturbed Hamiltonians eigenvalues problem
                for nqubit=1:numqubits
                    Hsmall=obj.circuits(nqubit).getH;
                    [Evect{1,nqubit},evals]=eigs(sparse(Hsmall),obj.lev1truncation(nqubit),'sr');
                    Htemp=Id;
                    Htemp(1,nqubit)={real(evals)};
                    if ~isempty(obj.lev2truncation)
                        H0lev2(1,nqubit)={real(evals)}; %copy single systems H0 for 2nd level h. diagonalisation
                    else
                        Hbig=Hbig+superkron(Htemp); %if no 2nd order h.d. is required, calculate the total H0
                        %i.e. the sum of the single circuits unperturbed
                        %Hamiltonians (corrected for capacitive/inductive
                        %loading)
                    end
                    if nqubit==circuitnum
                        %Get the operator for the matrix element
                        %calculation
                        if isequal(iopname(1),'I')
                            operator=obj.circuits(nqubit).getIops(iopname);
                        elseif isequal(iopname(1),'Q')
                            [~,Q]=obj.circuits(nqubit).getPQ(false,true);
                            Rinv=inv(obj.circuits(nqubit).rotation);
                            Rvec=Rinv(str2double(iopname(2)),:);
                            operator=ncon({Rvec',Q},{1,[-1 -2 1]});
                        end
                        %Project it on the truncated energy eigenbasis.
                        optemp=Id;
                        %optemp(1,nqubit)={ncon({conj(Evect{1,nqubit}),operator,Evect{1,nqubit}},{[1 -1],[1 2],[2,-2]})};
                        optemp(1,nqubit)={ncon({Evect{1,nqubit},operator,conj(Evect{1,nqubit})},{[1 -1],[1 2],[2,-2]})};
                    end
                end
                if ~isempty(obj.lev2truncation)
                    U=obj.lev2rotations(H0lev2); %calculate rotation mapping to the subsystems basis
                    Hbig=zeros(sizetot);
                    for nqubit=1:numqubits
                        Htemp=Id;
                        Htemp(1,nqubit)=H0lev2(1,nqubit);
                        Hbig=Hbig+obj.lev2truncate(Htemp,U); %apply rotation
                    end
                    operator=obj.lev2truncate(optemp,U); %redefine the operator 
                    %in the subsystem basis
                else
                    operator=superkron(optemp);
                end
                intindex=1;
                if ~isempty(Lmatg)
                    for Lind=find(triu(Lmatg,1))'%Lmatrow=1:length(Lmatg)
                        [Lmatrow,Lmatcol]=ind2sub(size(Lmatg),Lind);%for Lmatcol=Lmatrow+1:length(Lmatg)
                        %for every element of the inductance matrix,
                        %find the involved circuits
                        circn1=dict(Lmatrow).circuit;
                        circn2=dict(Lmatcol).circuit;
                        v1=Evect{1,circn1}; %get the unperturbed eigenvectors
                        v2=Evect{1,circn2};
                        %get the position of the two inductors to
                        %calculate the branch fluxes
                        inductn1=cellfun(@(x) isequal(x,dict(Lmatrow).inductor),{obj.circuits(circn1).inductors.names});
                        positions1=obj.circuits(dict(Lmatrow).circuit).inductors(inductn1).positions;
                        inductn2=cellfun(@(x) isequal(x,dict(Lmatcol).inductor),{obj.circuits(circn2).inductors.names});
                        positions2=obj.circuits(dict(Lmatcol).circuit).inductors(inductn2).positions;
                        [P1,~]=obj.circuits(circn1).getPQ;
                        [P2,~]=obj.circuits(circn2).getPQ;
                        Rinv1=inv(obj.circuits(circn1).rotation);
                        Rinv2=inv(obj.circuits(circn2).rotation);
                        if positions1(1)==positions1(2)
                            Pb1=squeeze(ncon({Rinv1(positions1(1),:),P1},{[-1,1],[-2 -3 1]}));
                        else
                            Pb1=squeeze(ncon({Rinv1(positions1(2),:),P1},{[-1,1],[-2 -3 1]}))-squeeze(ncon({Rinv1(positions1(1),:),P1},{[-1,1],[-2 -3 1]}));
                        end
                        if positions2(1)==positions2(2)
                            Pb2=squeeze(ncon({Rinv2(positions2(1),:),P2},{[-1,1],[-2 -3 1]}));
                        else
                            Pb2=squeeze(ncon({Rinv2(positions2(2),:),P2},{[-1,1],[-2 -3 1]}))-squeeze(ncon({Rinv2(positions2(1),:),P2},{[-1,1],[-2 -3 1]}));
                        end
                        %project the branch fluxes on the unperturbed
                        %Hamiltonian eigenvectors basis
                        H1=ncon({conj(v1),Pb1,v1},{[1 -1],[1 2],[2,-2]}); %Hint=H1*H2
                        H2=ncon({conj(v2),Pb2,v2},{[1 -1],[1 2],[2,-2]});
                        Hintern=Id;
                        if circn1==circn2
                            Hintern{1,circn1}=H1*H2; %mediated self-interaction term
                        else
                            Hintern{1,circn1}=H1;
                            Hintern{1,circn2}=H2;
                        end
                        %                                             Hintern{1,circn1}=H1{intindex};
                        %                                             Hintern{1,circn2}=H2{intindex};
                        %every interachtion term is of the form
                            %(L^-1)_ij*Phi_i*Phi_j with i!=j
                        if ~isempty(obj.lev2truncation)
                            Hinttemp=obj.lev2truncate(Hintern,U); %map interaction terms to subsystem basis
                            Hint(:,:,intindex)=Lmatg(Lmatrow,Lmatcol).*Hinttemp./obj.h*1e-9;
                            intindex=intindex+1;
                        else
                            Hint(:,:,intindex)=Lmatg(Lmatrow,Lmatcol)*superkron(Hintern)./obj.h*1e-9;
                            intindex=intindex+1;
                        end
                        %end
                    end
                end
                if ~isempty(Cmatg) && ~isempty(find(Cmatg,1))
                    for ind=find(triu(Cmatg,1))'
                        [node1,node2]=ind2sub([length(Cmatg),length(Cmatg)],ind);
                        circuitmarker=cumsum(sizes); %cumulative sum of the sequence of sizes of the capacitance
                        %matrices
                        circn1=find(node1<=circuitmarker,1);
                        circn2=find(node2<=circuitmarker,1);
                        v1=Evect{1,circn1}; %get the unperturbed eigenvectors
                        v2=Evect{1,circn2};
                        if circn1>1 %find the node number in the subcircuit
                            node1=node1-(circuitmarker(circn1-1));
                        end
                        if circn2>1
                            node2=node2-(circuitmarker(circn2-1));
                        end
                        [~,Q1]=obj.circuits(circn1).getPQ(false,true);
                        [~,Q2]=obj.circuits(circn2).getPQ(false,true);
                        Rinv1=obj.circuits(circn1).rotation';
                        Rinv2=obj.circuits(circn2).rotation';
                        Q1=squeeze(ncon({Rinv1(node1,:),Q1},{[-1,1],[-2 -3 1]}));
                        Q2=squeeze(ncon({Rinv2(node2,:),Q2},{[-1,1],[-2 -3 1]}));
                        %project the currents on the unperturbed
                        %Hamiltonian eigenvectors basis
                        H1=ncon({conj(v1),Q1,v1},{[1 -1],[1 2],[2,-2]}); %Hint=H1*H2
                        H2=ncon({conj(v2),Q2,v2},{[1 -1],[1 2],[2,-2]});
                        Hintern=Id;
                        if circn1==circn2
                            Hintern{1,circn1}=H1*H2; %mediated self-interaction term
                        else
                            Hintern{1,circn1}=H1;
                            Hintern{1,circn2}=H2;
                        end
                        %every interaction term is of the form
                            %(C^-1)_ij*Q_i*Q_j with i!=j
                        if ~isempty(obj.lev2truncation) %map interaction terms to subsystem basis
                            Hinttemp=obj.lev2truncate(Hintern,U);
                            Hint(:,:,intindex)=Cmatg(ind).*Hinttemp./obj.h*1e-9;
                            intindex=intindex+1;
                        else
                            Hint(:,:,intindex)=Cmatg(ind)*superkron(Hintern)./obj.h*1e-9;
                            intindex=intindex+1;
                        end
                    end
                end
                Htot=Hbig+sum(Hint,3).';
                opts.maxit=500;
                [evect,evals]=eigs(sparse(Htot),max(initstate,finstate),'sr',opts);
                [~,order]=sort(real(diag(evals)));
                initvec=evect(:,order(initstate));
                initvec=obj.normeigenvectors(initvec);
                finvec=evect(:,order(finstate));
                finvec=obj.normeigenvectors(finvec);
                out=finvec'*operator*initvec*1e9;
                if initstate==finstate
                    out=real(out);
                end
                if isequal(iopname(1),'Q')
                    out=out/obj.circuits(circuitnum).parammat.Cmat(str2double(iopname(2)),str2double(iopname(2)));
                end
                obj.resetloadparammats; %reset capacitive and inductive matrices
            end
        end
        
        %% Calculates the energy spectrum of the coupled system as a function of one or more varying parameters
        % SWEEPOP Calculates the matrix element of a current/charge
        % operator between two states, as a function of a varying
        % parameter. The first input is the name of the previously defined
        % sweep, the second is the index of the circuit affected, the third
        % is the name of the operator. Inputs 4 and 5 are the eigenstates
        % to be used in the matrix element (in order of ascending energy,
        % with 1 for the ground state). Input 6 is the 'Plot' option, which
        % turns on a plot of the result if input 7 is set to true.
        function out=sweepOp(obj,sweepname,circuitnum,iopname,initstate,finstate,plotopt,optionp)
            numqubits=length(obj.circuitnames);
            if length(obj.interactions)<1
                fprintf("Define pairs of interacting qubits\n");
%             elseif length(obj.minductances)+length(obj.couplingcap)<length(obj.interpairs)
%                 fprintf("Define mutual inductances/shared capacitances values\n");
%             elseif length(obj.coupledinductors)+length(obj.couplednodes)<length(obj.interpairs)
%                 fprintf("Define coupling branch currents or node charges\n");
            elseif length(obj.lev1truncation)<numqubits
                fprintf("Define upper level truncation values\n");
            elseif nargin < 6
                fprintf("Specify sweepname, circuit index, operator name, initial and final states\n");
            elseif isempty(obj.sweeps)
                fprintf("Define a sweep first using the function paramsweep or fluxsweep\n");
            else
                %
                if any(cellfun(@(x) isequal(x,'Ind'),{obj.interactions.ctype}))
                    [dict,Lmatg]=obj.buildinductancemat(); %get global branch inductance matrix
                    Lmatg=inv(Lmatg); %invert the inductance matrix
                else
                    Lmatg=[];
                end
                if any(cellfun(@(x) isequal(x,'Cap'),{obj.interactions.ctype}))
                    [~,Cmatg,sizes]=obj.buildcapacitancemat(); %get inverse global node capacitance matrix
                else
                    Cmatg=[];
                end
                obj.setloadparammats;
                numinter=length(find(triu(Lmatg,1)))+length(find(triu(Cmatg,1))); %Number of interaction terms
                %it equals half the number of off diagonal terms in the
                %inductance matrix plus half the number of block-off diagonal
                %terms in the capacitance matrix 
                H1=cell(1,numinter);
                H2=cell(1,numinter);
                numsweptpar=0;
                toggle=0;
                %check that the specified sweep has been previously defined
                %(through paramsweep() or fluxsweep())
                for sweepn=1:length(obj.sweeps)
                    if isequal(obj.sweeps(sweepn).sweepname,sweepname)
                        numsweptpar=numsweptpar+1; %if a sweep with the desired name is found, increase numsweptpar by 1
                        if strcmp(obj.sweeps(sweepn).parameter(1),'L') || strcmp(obj.sweeps(sweepn).parameter(1),'C') || isequal(obj.sweeps(sweepn).parameter,'alpha') 
                            %if the sweep changes the the flux/current operators, turn toggle on,
                            toggle=1;
                        end
                    end
                end
                if numsweptpar==0
                    fprintf("Specify an existent sweep name\n");
                else
                    %copy the sweep data
                    sweep=struct('sweepname',sweepname,'qubitnum',cell(1,numsweptpar),'sweeptype',cell(1,numsweptpar),'parameter',...
                        cell(1,numsweptpar),'range',cell(1,numsweptpar));
                    param=1;
                    for sweepn=1:length(obj.sweeps)
                        if isequal(obj.sweeps(sweepn).sweepname,sweepname)
                            sweep(param)=obj.sweeps(sweepn); % Only copy the sweeps with the right name
                            param=param+1;
                        end
                    end
                    sweeplengths=cellfun(@length,{sweep.range});
                    if all(sweeplengths==sweeplengths(1)) %checks that all swept parameters take the same number of values
                        if length(initstate)==1 && length(finstate)==1
                            out=zeros(1,length(sweep(1).range));
                        elseif max(length(initstate),length(finstate))>1 && min(length(initstate),length(finstate))==1
                            out=zeros(max(length(initstate),length(finstate)),length(sweep(1).range));
                        elseif length(initstate)==length(finstate) && length(initstate)>1
                            out=zeros(length(initstate),length(sweep(1).range));
                        else
                            fprintf("The number of initial and final states need to be equal, unless one of these two numbers is 1.\n")
                        end
                        if isequal(sweep(1).sweeptype,'param') %parameter sweep: requires calculating the full hamiltonian every time
                            paramold=zeros(1,numsweptpar); %copy the original value of the swept parameter
                            for paramn=1:numsweptpar
                                paramold(1,paramn)=obj.circuits(sweep(paramn).qubitnum).parameters.(sweep(paramn).parameter);
                            end
                            f=waitbar(1/length(sweep(1).range),strcat('Sweep progress:',num2str(100/length(sweep(1).range),2),'%'));
                            for iter=1:length(sweep(1).range) %for every point in the sweep
                                waitbar(iter/length(sweep(1).range),f,strcat('Sweep progress:',num2str(100*iter/length(sweep(1).range),2),'%'));
                                for paramn=1:numsweptpar %change qubit properties
                                    obj.circuits(sweep(paramn).qubitnum).parameters.(sweep(paramn).parameter)=sweep(paramn).range(iter);
                                end
                                if toggle %recalculate loaded matrices to account for a change in L or C
                                    obj.resetloadparammats;
                                    if any(cellfun(@(x) isequal(x,'Ind'),{obj.interactions.ctype}))
                                        [dict,Lmatg]=obj.buildinductancemat(); %get global branch inductance matrix
                                        Lmatg=inv(Lmatg); %invert the inductance matrix
                                    else
                                        Lmatg=[];
                                    end
                                    if any(cellfun(@(x) isequal(x,'Cap'),{obj.interactions.ctype}))
                                        [~,Cmatg,sizes]=obj.buildcapacitancemat(); %get inverse global node capacitance matrix
                                    else
                                        Cmatg=[];
                                    end
                                    obj.setloadparammats;
                                end
                                if iter==1
                                    %initialise cells at the first
                                    %iteration
                                    if isempty(obj.lev2truncation)
                                        Hbig=zeros(prod(obj.lev1truncation)); %Sum of the single system hamiltonians
                                        Hint=zeros(prod(obj.lev1truncation),prod(obj.lev1truncation),numinter); %Interaction hamiltonians
                                    else
                                        if isempty(obj.lev3truncation)
                                            dims=obj.lev2truncation;
                                        else
                                            dims=obj.lev3truncation;
                                        end
                                        subsnum=size(dims,1);
                                        for subind=1:subsnum
                                            if subind==1
                                                sizetot=dims{subind,:}{2};
                                            else
                                                sizetot=sizetot*dims{subind,:}{2};
                                            end
                                        end
                                        Hint=zeros(sizetot,sizetot,numinter);
                                    end
                                    Evect=cell(1,numqubits);
                                    H0lev2=cell(1,numqubits); %Unperturbed Hamiltonian for 2nd level hierarchical diagonalisation
                                    Id=cellfun(@eye,num2cell(obj.lev1truncation),'UniformOutput',0);
                                    P=cell(1,numqubits);
                                    Q=cell(1,numqubits);
                                    %calculate operators necessary for
                                    %interaction terms
                                    if ~isempty(Lmatg) %if there are iductive couplings
                                        for nqubit=1:numqubits %node fluxes
                                            [Ptemp,~]=obj.circuits(nqubit).getPQ;
                                            Rinv=inv(obj.circuits(nqubit).rotation);
                                            P(1,nqubit)={ncon({Rinv,Ptemp},{[-3,1],[-1 -2 1]})};
                                        end
                                    end
                                    if ~isempty(Cmatg) && ~isempty(find(Cmatg,1))
                                        for nqubit=1:numqubits %node charges
                                            [~,Qtemp]=obj.circuits(nqubit).getPQ;
                                            Rinv=obj.circuits(nqubit).rotation';
                                            Q(1,nqubit)={ncon({Rinv,Qtemp},{[-3,1],[-1 -2 1]})};
                                        end
                                    end
                                    %solve unperturbed Hamiltonian problems
                                    for nqubit=1:numqubits
                                        Hsmall=obj.circuits(nqubit).getH;
                                        [Evect{1,nqubit},evals]=eigs(sparse(Hsmall),obj.lev1truncation(nqubit),'SR');
                                        Htemp=Id;
                                        Htemp(1,nqubit)={real(evals)};
                                        if ~isempty(obj.lev2truncation)
                                            H0lev2(1,nqubit)={real(evals)}; %copy single systems H0 for 2nd level h. diagonalisation
                                        else
                                            Hbig=Hbig+superkron(Htemp); %if no 2nd order h.d. is required, calculate the total H0
                                        end
                                        if nqubit==circuitnum
                                            %Get the operator for the matrix element
                                            %calculation
                                            if isequal(iopname(1),'I')
                                                operator=obj.circuits(nqubit).getIops(iopname);
                                            elseif isequal(iopname(1),'Q')
                                                [~,Q]=obj.circuits(nqubit).getPQ;
                                                Rinv=inv(obj.circuits(nqubit).rotation);
                                                Rvec=Rinv(str2double(iopname(2)),:);
                                                operator=ncon({Rvec',Q},{1,[-1 -2 1]});
                                            end
                                            %Project it on the truncated energy eigenbasis.
                                            optemp=Id;
                                            optemp(1,nqubit)={ncon({Evect{1,nqubit},operator,conj(Evect{1,nqubit})},{[1 -1],[1 2],[2,-2]})};
                                        end
                                    end
                                    if ~isempty(obj.lev2truncation)
                                        U=obj.lev2rotations(H0lev2);
                                        Hbig=zeros(sizetot);
                                        for nqubit=1:numqubits
                                            Htemp=Id;
                                            Htemp(1,nqubit)=H0lev2(1,nqubit);
                                            Hbig=Hbig+obj.lev2truncate(Htemp,U);
                                        end
                                        operator=obj.lev2truncate(optemp,U); %redefine the operator 
                                        %in the subsystem basis
                                    else
                                        operator=superkron(optemp);
                                    end
                                    %find the interaction terms
                                    intindex=1;
                                    if ~isempty(Lmatg)
                                        for Lind=find(triu(Lmatg,1))'%Lmatrow=1:length(Lmatg)
                                            [Lmatrow,Lmatcol]=ind2sub(size(Lmatg),Lind);%for Lmatcol=Lmatrow+1:length(Lmatg)
                                            %for every element of the inductance matrix,
                                            %find the involved circuits
                                            circn1=dict(Lmatrow).circuit;
                                            circn2=dict(Lmatcol).circuit;

                                            v1=Evect{1,circn1}; %get the unperturbed eigenvectors
                                            inductn1=cellfun(@(x) isequal(x,dict(Lmatrow).inductor),{obj.circuits(circn1).inductors.names});
                                            positions1=obj.circuits(dict(Lmatrow).circuit).inductors(inductn1).positions;
                                            if positions1(1)==positions1(2)
                                                Pb1=P{circn1}(:,:,positions1(1));
                                            else
                                                Pb1=P{circn1}(:,:,positions1(2))-P{circn1}(:,:,positions1(1));
                                            end
                                            H1(intindex)={ncon({conj(v1),Pb1,v1},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2

                                            v2=Evect{1,circn2}; %get the unperturbed eigenvectors
                                            inductn2=cellfun(@(x) isequal(x,dict(Lmatcol).inductor),{obj.circuits(circn2).inductors.names});
                                            positions2=obj.circuits(dict(Lmatcol).circuit).inductors(inductn2).positions;
                                            if positions2(1)==positions2(2)
                                                Pb2=P{circn2}(:,:,positions2(1));
                                            else
                                                Pb2=P{circn2}(:,:,positions2(2))-P{circn2}(:,:,positions2(1));
                                            end
                                            H2(intindex)={ncon({conj(v2),Pb2,v2},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2

                                            Hintern=Id;
                                            if circn1==circn2
                                                Hintern{1,circn1}=H1{intindex}*H2{intindex}; %mediated self-interaction term
                                            else
                                                Hintern{1,circn1}=H1{intindex};
                                                Hintern{1,circn2}=H2{intindex};
                                            end
                                            if ~isempty(obj.lev2truncation)
                                                Hinttemp=obj.lev2truncate(Hintern,U);
                                                Hint(:,:,intindex)=Lmatg(Lmatrow,Lmatcol).*Hinttemp./obj.h*1e-9;
                                                intindex=intindex+1;
                                            else
                                                %every interachtion term is of the form
                                                %(L^-1)_ij*Phi_i*Phi_j with i!=j
                                                Hint(:,:,intindex)=Lmatg(Lmatrow,Lmatcol)*superkron(Hintern)./obj.h*1e-9;
                                                intindex=intindex+1;
                                            end
                                            %end
                                        end
                                    end
                                    if ~isempty(Cmatg) && ~isempty(find(Cmatg,1))
                                        for ind=find(triu(Cmatg,1))'
                                            %for every element of the inverse matrix,
                                            %find the involved circuits
                                            [node1,node2]=ind2sub([length(Cmatg),length(Cmatg)],ind);
                                            circuitmarker=cumsum(sizes); %cumulative sum of the sequence of sizes of the capacitance
                                            %matrices
                                            circn1=find(node1<=circuitmarker,1);
                                            circn2=find(node2<=circuitmarker,1);
                                            if circn1>1 %find the node index in the subsystem
                                                node1=node1-(circuitmarker(circn1-1));
                                            end
                                            
                                            v1=Evect{1,circn1}; %get the unperturbed eigenvectors
                                            Q1=Q{circn1}(:,:,node1);
                                            H1(intindex)={ncon({conj(v1),Q1,v1},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                            %H1(intindex)={ncon({v1,Q1,conj(v1)},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2

                                            v2=Evect{1,circn2}; %get the unperturbed eigenvectors
                                            Q2=Q{circn2}(:,:,node2);
                                            H2(intindex)={ncon({conj(v2),Q2,v2},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                            
                                            Hintern=Id;
                                            if circn1==circn2
                                                Hintern{1,circn1}=H1{intindex}*H2{intindex}; %mediated self-interaction term
                                            else
                                                Hintern{1,circn1}=H1{intindex};
                                                Hintern{1,circn2}=H2{intindex};
                                            end
                                            if ~isempty(obj.lev2truncation)
                                                Hinttemp=obj.lev2truncate(Hintern,U);
                                                Hint(:,:,intindex)=Cmatg(ind).*Hinttemp./obj.h*1e-9;
                                                intindex=intindex+1;
                                            else
                                                %every interachtion term is of the form
                                                %(C^-1)_ij*Q_i*Q_j with i!=j
                                                Hint(:,:,intindex)=Cmatg(ind)*superkron(Hintern)./obj.h*1e-9;
                                                intindex=intindex+1;
                                            end
                                        end
                                    end
                                else
                                    %repeat the previous passages with the
                                    %qubits affected by the sweep
                                    if toggle %if the capacitance and inductance matrices have changed,
                                        %recalculate the canonical variables operators
                                        if ~isempty(Lmatg) %if there are iductive couplings
                                            [~,Lmatg]=obj.buildinductancemat(); %recalculate global branch inductance matrix
                                            Lmatg=inv(Lmatg); %invert the inductance matrix
                                            for nqubit=1:unique([sweep.qubitnum]) %node fluxes
                                                [Ptemp,~]=obj.circuits(nqubit).getPQ;
                                                Rinv=inv(obj.circuits(nqubit).rotation);
                                                P(1,nqubit)={ncon({Rinv,Ptemp},{[-3,1],[-1 -2 1]})};
                                            end
                                        end
                                        if ~isempty(Cmatg) && ~isempty(find(Cmatg,1))
                                            [~,Cmatg,~]=obj.buildcapacitancemat(); %recalculate inverse global node capacitance matrix
                                            for nqubit=1:unique([sweep.qubitnum]) %node charges
                                                [~,Qtemp]=obj.circuits(nqubit).getPQ;
                                                Rinv=obj.circuits(nqubit).rotation';
                                                Q(1,nqubit)={ncon({Rinv,Qtemp},{[-3,1],[-1 -2 1]})};
                                            end
                                        end 
                                    end
                                    for nqubit=unique([sweep.qubitnum])
                                        Hsmall=obj.circuits(nqubit).getH;
                                        [Evect{1,nqubit},evals]=eigs(sparse(Hsmall),obj.lev1truncation(nqubit),'SR');
                                        Htemp=Id;
                                        Htemp(1,nqubit)={real(evals)};
                                        if ~isempty(obj.lev2truncation)
                                            H0lev2(1,nqubit)={real(evals)}; %copy single systems H0 for 2nd level h. diagonalisation
                                        else
                                            Hbig=Hbig+superkron(Htemp); %if no 2nd order h.d. is required, calculate the total H0
                                        end
                                        if nqubit==circuitnum
                                            %Get the operator for the matrix element
                                            %calculation
                                            if isequal(iopname(1),'I')
                                                operator=obj.circuits(nqubit).getIops(iopname);
                                            elseif isequal(iopname(1),'Q')
                                                [~,Q]=obj.circuits(nqubit).getPQ;
                                                Rinv=inv(obj.circuits(nqubit).rotation);
                                                Rvec=Rinv(str2double(iopname(2)),:);
                                                operator=ncon({Rvec',Q},{1,[-1 -2 1]});
                                            end
                                            %Project it on the truncated energy eigenbasis.
                                            optemp=Id;
                                            optemp(1,nqubit)={ncon({Evect{1,nqubit},operator,conj(Evect{1,nqubit})},{[1 -1],[1 2],[2,-2]})};
                                            operator=superkron(optemp);
                                        end
                                    end
                                    if ~isempty(obj.lev2truncation)
                                        U=obj.lev2rotations(H0lev2);
                                        Hbig=zeros(sizetot);
                                        for nqubit=1:numqubits
                                            Htemp=Id;
                                            Htemp(1,nqubit)=H0lev2(1,nqubit);
                                            Hbig=Hbig+obj.lev2truncate(Htemp,U);
                                        end
                                        operator=obj.lev2truncate(optemp,U); %redefine the operator 
                                        %in the subsystem basis
                                    end
                                    %find the interaction terms
                                    intindex=1;
                                    if ~isempty(Lmatg)
                                        for Lind=find(triu(Lmatg,1))'%Lmatrow=1:length(Lmatg)
                                            [Lmatrow,Lmatcol]=ind2sub(size(Lmatg),Lind);%for Lmatcol=Lmatrow+1:length(Lmatg)
                                            %for every element of the inductance matrix,
                                            %find the involved circuits
                                            circn1=dict(Lmatrow).circuit;
                                            circn2=dict(Lmatcol).circuit;
                                            if length(unique([sweep.qubitnum]))>1 %check if the first and the second 
                                                %circuit are involved
                                                %in the sweep
                                                toggle1=contains(unique([sweep.qubitnum]),circn1);
                                                toggle2=contains(unique([sweep.qubitnum]),circn2);
                                            else
                                                toggle1=isequal(unique([sweep.qubitnum]),circn1);
                                                toggle2=isequal(unique([sweep.qubitnum]),circn2);
                                            end
                                            if toggle1 %if they are involved in the sweep,
                                                %recalculate the
                                                %expectation values of
                                                %the branch fluxes
                                                v1=Evect{1,circn1}; %get the unperturbed eigenvectors
                                                inductn1=cellfun(@(x) isequal(x,dict(Lmatrow).inductor),{obj.circuits(circn1).inductors.names});
                                                positions1=obj.circuits(dict(Lmatrow).circuit).inductors(inductn1).positions;
                                                if positions1(1)==positions1(2)
                                                    Pb1=P{circn1}(:,:,positions1(1));
                                                else
                                                    Pb1=P{circn1}(:,:,positions1(2))-P{circn1}(:,:,positions1(1));
                                                end
                                                H1(intindex)={ncon({conj(v1),Pb1,v1},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                            end
                                            if toggle2
                                                v2=Evect{1,circn2}; %get the unperturbed eigenvectors
                                                inductn2=cellfun(@(x) isequal(x,dict(Lmatcol).inductor),{obj.circuits(circn2).inductors.names});
                                                positions2=obj.circuits(dict(Lmatcol).circuit).inductors(inductn2).positions;
                                                if positions2(1)==positions2(2)
                                                    Pb2=P{circn2}(:,:,positions2(1));
                                                else
                                                    Pb2=P{circn2}(:,:,positions2(2))-P{circn2}(:,:,positions2(1));
                                                end
                                                H2(intindex)={ncon({conj(v2),Pb2,v2},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                            end
                                            Hintern=Id;
                                            if circn1==circn2
                                                Hintern{1,circn1}=H1{intindex}*H2{intindex}; %mediated self-interaction term
                                            else
                                                Hintern{1,circn1}=H1{intindex};
                                                Hintern{1,circn2}=H2{intindex};
                                            end
                                            if ~isempty(obj.lev2truncation)
                                                Hinttemp=obj.lev2truncate(Hintern,U);
                                                Hint(:,:,intindex)=Lmatg(Lmatrow,Lmatcol).*Hinttemp./obj.h*1e-9;
                                                intindex=intindex+1;
                                            else
                                                %every interachtion term is of the form
                                                %(L^-1)_ij*Phi_i*Phi_j with i!=j
                                                Hint(:,:,intindex)=Lmatg(Lmatrow,Lmatcol)*superkron(Hintern)./obj.h*1e-9;
                                                intindex=intindex+1;
                                            end
                                            %end
                                        end
                                    end
                                    if ~isempty(Cmatg) && ~isempty(find(Cmatg,1))
                                        for ind=find(triu(Cmatg,1))'
                                            %for every element of the inverse matrix,
                                            %find the involved circuits
                                            [node1,node2]=ind2sub([length(Cmatg),length(Cmatg)],ind);
                                            circuitmarker=cumsum(sizes); %cumulative sum of the sequence of sizes of the capacitance
                                            %matrices
                                            circn1=find(node1<=circuitmarker,1);
                                            circn2=find(node2<=circuitmarker,1);
                                            if circn1>1 %find the node index in the subsystem
                                                node1=node1-(circuitmarker(circn1-1));
                                            end
                                            if circn2>1
                                                node2=node2-(circuitmarker(circn2-1));
                                            end
                                            if length(unique([sweep.qubitnum]))>1 %check if the first and the second 
                                                %circuit are involved
                                                %in the sweep
                                                toggle1=contains(unique([sweep.qubitnum]),circn1);
                                                toggle2=contains(unique([sweep.qubitnum]),circn2);
                                            else
                                                toggle1=isequal(unique([sweep.qubitnum]),circn1);
                                                toggle2=isequal(unique([sweep.qubitnum]),circn2);
                                            end
                                            if iter==1 || toggle1 % if the circuits are involved in the sweep
                                                %recalculate the
                                                %expectation values of the
                                                %charge operators
                                                v1=Evect{1,circn1}; %get the unperturbed eigenvectors
                                                Q1=Q{circn1}(:,:,node1);
                                                H1(intindex)={ncon({conj(v1),Q1,v1},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                            end
                                            if iter==1 || toggle2
                                                v2=Evect{1,circn2}; %get the unperturbed eigenvectors
                                                Q2=Q{circn2}(:,:,node2);
                                                H2(intindex)={ncon({conj(v2),Q2,v2},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                            end
                                            Hintern=Id;
                                            if circn1==circn2
                                                Hintern{1,circn1}=H1{intindex}*H2{intindex}; %mediated self-interaction term
                                            else
                                                Hintern{1,circn1}=H1{intindex};
                                                Hintern{1,circn2}=H2{intindex};
                                            end
                                            if ~isempty(obj.lev2truncation)
                                                Hinttemp=obj.lev2truncate(Hintern,U);
                                                Hint(:,:,intindex)=Cmatg(ind).*Hinttemp./obj.h*1e-9;
                                                intindex=intindex+1;
                                            else
                                                %every interachtion term is of the form
                                                %(C^-1)_ij*Q_i*Q_j with i!=j
                                                Hint(:,:,intindex)=Cmatg(ind)*superkron(Hintern)./obj.h*1e-9;
                                                intindex=intindex+1;
                                            end
                                        end
                                    end
                                end
                                %sum unperturbed and perturbed Hamiltonians
                                Htot=sum(Hbig,3)+sum(Hint,3).';
                                opts.maxit=500;
                                [evect,evals]=eigs(sparse(Htot),max(cat(2,initstate,finstate)),'SA',opts);
                                [~,order]=sort(real(diag(evals)));
                                if length(initstate)==1 && length(finstate)==1
                                    initvec=obj.normeigenvectors(evect(:,order(initstate)));
                                    finvec=obj.normeigenvectors(evect(:,order(finstate)));
                                    if initstate==finstate
                                        out(iter)=real(finvec'*operator*initvec*1e9);
                                    else
                                        out(iter)=finvec'*operator*initvec*1e9;
                                    end
                                    if isequal(iopname(1),'Q')
                                        out(1,iter)=out(1,iter)/obj.circuits(circuitnum).parammat.Cmat(str2double(iopname(2)),str2double(iopname(2)));
                                    end
                                elseif length(initstate)>1 && length(finstate)==1
                                    finvec=obj.normeigenvectors(evect(:,order(finstate)));
                                    initvec=evect(:,order(initstate));
                                    for inits=1:length(initstate)
                                        invs=obj.normeigenvectors(initvec(:,inits));
                                        out(inits,iter)=finvec'*operator*invs*1e9;
                                    end
                                    if isequal(iopname(1),'Q')
                                        out(:,iter)=out(:,iter)/obj.circuits(circuitnum).parammat.Cmat(str2double(iopname(2)),str2double(iopname(2)));
                                    end
                                elseif length(finstate)>1 && length(initstate)==1
                                    initvec=obj.normeigenvectors(evect(:,order(initstate)));
                                    finvec=evect(:,order(finstate));
                                    for finits=1:length(finstate)
                                        finv=obj.normeigenvectors(finvec(:,finits));
                                        out(finits,iter)=finv'*operator*initvec*1e9;
                                    end
                                    if isequal(iopname(1),'Q')
                                        out(:,iter)=out(:,iter)/obj.circuits(circuitnum).parammat.Cmat(str2double(iopname(2)),str2double(iopname(2)));
                                    end
                                elseif length(initstate)==length(finstate) && length(initstate)>1
                                    initvec=evect(:,order(initstate));
                                    finvec=evect(:,order(finstate));
                                    for inits=1:length(initstate)
                                        invs=obj.normeigenvectors(initvec(:,inits));
                                        finv=obj.normeigenvectors(finvec(:,inits));
                                        out(inits,iter)=finv'*operator*invs*1e9;
                                    end
                                    if isequal(iopname(1),'Q')
                                        out(:,iter)=out(:,iter)/obj.circuits(circuitnum).parammat.Cmat(str2double(iopname(2)),str2double(iopname(2)));
                                    end
                                end
                            end
                            obj.resetloadparammats; %restore parammats
                            for paramn=1:numsweptpar %restore original param values
                                obj.circuits(sweep(paramn).qubitnum).parameters.(sweep(paramn).parameter)=paramold(1,paramn);
                            end
                        elseif isequal(sweep(1).sweeptype,'flux') %flux sweep, only the bias matrix needs to be recalculated
                            f=waitbar(1/length(sweep(1).range),strcat('Sweep progress:',num2str(100/length(sweep(1).range),2),'%'));
                            for iter=1:length(sweep(1).range) %for every point in the sweep
                                waitbar(iter/length(sweep(1).range),f,strcat('Sweep progress:',num2str(100*iter/length(sweep(1).range),2),'%'));
                                if iter==1
                                    %initialise cells at the first
                                    %iteration
                                    Id=cellfun(@eye,num2cell(obj.lev1truncation),'UniformOutput',0);
                                    if isempty(obj.lev2truncation)
                                        Hbig=zeros(prod(obj.lev1truncation),prod(obj.lev1truncation),numqubits); %Single system hamiltonians
                                        Hint=zeros(prod(obj.lev1truncation),prod(obj.lev1truncation),numinter); %Interaction hamiltonians
                                    else
                                        subsnum=size(obj.lev2truncation,1);
                                        for subind=1:subsnum
                                            if subind==1
                                                sizetot=obj.lev2truncation{subind,:}{2};
                                            else
                                                sizetot=sizetot*obj.lev2truncation{subind,:}{2};
                                            end
                                        end
                                        Hint=zeros(sizetot,sizetot,numinter);
                                    end
                                    HLC=cell(1,numqubits);
                                    Dmat=cell(1,numqubits);
                                    Evect=cell(1,numqubits);
                                    H0lev2=cell(1,numqubits); %Unperturbed Hamiltonian for 2nd level hierarchical diagonalisation
                                    P=cell(1,numqubits);
                                    Q=cell(1,numqubits);
                                    %calculate operators necessary for
                                    %interaction terms
                                    if ~isempty(Lmatg) %if there are iductive couplings
                                        for nqubit=1:numqubits %node fluxes
                                            [Ptemp,~]=obj.circuits(nqubit).getPQ;
                                            Rinv=inv(obj.circuits(nqubit).rotation);
                                            P(1,nqubit)={ncon({Rinv,Ptemp},{[-3,1],[-1 -2 1]})};
                                        end
                                    end
                                    if ~isempty(Cmatg) && ~isempty(find(Cmatg,1))
                                        for nqubit=1:numqubits %node charges
                                            [~,Qtemp]=obj.circuits(nqubit).getPQ;
                                            Rinv=obj.circuits(nqubit).rotation';
                                            Q(1,nqubit)={ncon({Rinv,Qtemp},{[-3,1],[-1 -2 1]})};
                                        end
                                    end
                                    nummodes=zeros(1,numqubits);
                                    for nqubit=1:numqubits
                                        %unperturbed Hamiltonians
                                        nummodes(nqubit)=length(obj.circuits(nqubit).modes);
                                        HLC(1,nqubit)={obj.circuits(nqubit).getHLC};
                                        Dmat(1,nqubit)={obj.circuits(nqubit).getDops};
                                        EJ=obj.circuits(nqubit).parammat.Ejmat;
                                        HJ=sum(sum(EJ)).*eye(length(HLC{1,nqubit}));
                                        if any(find([sweep.qubitnum]==nqubit))
                                            biasmatrix=obj.circuits(nqubit).getBiasm(sweep([sweep.qubitnum]==nqubit),1); % pass the value from the sweep only to the qubits
                                        else                                                      % affected by the sweep
                                            biasmatrix=obj.circuits(nqubit).getBiasm();
                                        end
                                        for ind=find(EJ)'
                                            [Dmatrow,Dmatcolumn]=ind2sub(size(EJ,1),ind);
                                            HJ=HJ-EJ(Dmatrow,Dmatcolumn)/2.*(biasmatrix(Dmatrow,Dmatcolumn).*Dmat{1,nqubit}{Dmatrow,Dmatcolumn}+...
                                                conj(biasmatrix(Dmatrow,Dmatcolumn)).*Dmat{1,nqubit}{Dmatrow,Dmatcolumn}');
%                                                 end
%                                             end
                                        end
                                        HJ=HJ./obj.h*1e-9;
                                        [Evect{1,nqubit},evals]=eigs(sparse(HLC{1,nqubit}+HJ),obj.lev1truncation(nqubit),'SR');
                                        Htemp=Id;
                                        Htemp(1,nqubit)={real(evals)};
                                        if ~isempty(obj.lev2truncation)
                                            H0lev2(1,nqubit)={real(evals)}; %copy single systems H0 for 2nd level h. diagonalisation
                                        else
                                            Hbig(:,:,nqubit)=superkron(Htemp); %if no 2nd order h.d. is required, calculate the total H0
                                        end
                                        if nqubit==circuitnum
                                            %Get the operator for the matrix element
                                            %calculation
                                            if isequal(iopname(1),'I')
                                                operator=obj.circuits(nqubit).getIops(iopname);
                                            elseif isequal(iopname(1),'Q')
                                                [~,Q]=obj.circuits(nqubit).getPQ;
                                                Rinv=inv(obj.circuits(nqubit).rotation);
                                                Rvec=Rinv(str2double(iopname(2)),:);
                                                operator=ncon({Rvec',Q},{1,[-1 -2 1]});
                                            end
                                            %Project it on the truncated energy eigenbasis.
                                            optemp=Id;
                                            optemp(1,nqubit)={ncon({Evect{1,nqubit},operator,conj(Evect{1,nqubit})},{[1 -1],[1 2],[2,-2]})};
                                        end
                                    end
                                    if ~isempty(obj.lev2truncation)
                                        U=obj.lev2rotations(H0lev2);
                                        %                     Hbig=obj.lev2truncate(H0lev2,U);
                                        Hbig=zeros(sizetot);
                                        for nqubit=1:numqubits
                                            Htemp=Id;
                                            Htemp(1,nqubit)=H0lev2(1,nqubit);
                                            Hbig=Hbig+obj.lev2truncate(Htemp,U);
                                        end
                                        operator=obj.lev2truncate(optemp,U); %redefine the operator 
                                        %in the subsystem basis
                                    else
                                        operator=superkron(optemp);
                                    end
                                else
                                    for nqubit=unique([sweep.qubitnum]) %update only the unperturbed Hamiltonians
                                        %relative to qubits involved in the sweep
                                        EJ=obj.circuits(nqubit).parammat.Ejmat;
                                        HJ=sum(sum(EJ)).*eye(length(HLC{1,nqubit}));
                                        %update the biases values in the
                                        %Josephson terms
                                        biasmatrix=obj.circuits(nqubit).getBiasm(sweep([sweep.qubitnum]==nqubit),iter);
                                        for ind=find(EJ)'
                                            [Dmatrow,Dmatcolumn]=ind2sub(size(EJ,1),ind);
                                            HJ=HJ-EJ(Dmatrow,Dmatcolumn)/2.*(biasmatrix(Dmatrow,Dmatcolumn).*Dmat{1,nqubit}{Dmatrow,Dmatcolumn}+...
                                                conj(biasmatrix(Dmatrow,Dmatcolumn)).*Dmat{1,nqubit}{Dmatrow,Dmatcolumn}');
%                                                 end
%                                             end
                                        end
                                        HJ=HJ./obj.h*1e-9;
                                        [Evect{1,nqubit},evals]=eigs(sparse(HLC{1,nqubit}+HJ),obj.lev1truncation(nqubit),'SR');
                                        Htemp=Id;
                                        Htemp(1,nqubit)={real(evals)};
                                        if ~isempty(obj.lev2truncation)
                                            H0lev2(1,nqubit)={real(evals)}; %copy single systems H0 for 2nd level h. diagonalisation
                                        else
                                            Hbig(:,:,nqubit)=superkron(Htemp); %if no 2nd order h.d. is required, calculate the total H0
                                        end
                                        if nqubit==circuitnum
                                            %Get the operator for the matrix element
                                            %calculation
                                            if isequal(iopname(1),'I')
                                                operator=obj.circuits(nqubit).getIops(iopname);
                                            elseif isequal(iopname(1),'Q')
                                                [~,Q]=obj.circuits(nqubit).getPQ;
                                                Rinv=inv(obj.circuits(nqubit).rotation);
                                                Rvec=Rinv(str2double(iopname(2)),:);
                                                operator=ncon({Rvec',Q},{1,[-1 -2 1]});
                                            end
                                            %Project it on the truncated energy eigenbasis.
                                            optemp=Id;
                                            optemp(1,nqubit)={ncon({Evect{1,nqubit},operator,conj(Evect{1,nqubit})},{[1 -1],[1 2],[2,-2]})};
                                            operator=superkron(optemp);
                                        end
                                    end
                                    if ~isempty(obj.lev2truncation)
                                        U=obj.lev2rotations(H0lev2);
                                        Hbig=zeros(sizetot);
                                        for nqubit=1:numqubits
                                            Htemp=Id;
                                            Htemp(1,nqubit)=H0lev2(1,nqubit);
                                            Hbig=Hbig+obj.lev2truncate(Htemp,U);
                                        end
                                        operator=obj.lev2truncate(optemp,U); %redefine the operator 
                                        %in the subsystem basis
                                    end
                                end
                                intindex=1;
                                if ~isempty(Lmatg)
                                    for Lind=find(triu(Lmatg,1))'%Lmatrow=1:length(Lmatg)
                                        [Lmatrow,Lmatcol]=ind2sub(size(Lmatg),Lind);%for Lmatcol=Lmatrow+1:length(Lmatg)
                                        %for every element of the inductance matrix,
                                        %find the involved circuits
                                        circn1=dict(Lmatrow).circuit;
                                        circn2=dict(Lmatcol).circuit;
                                        if length(unique([sweep.qubitnum]))>1 %check if the circuits are involved in the sweep
                                            toggle1=~isempty(find([sweep.qubitnum]==circn1, 1));
                                            toggle2=~isempty(find([sweep.qubitnum]==circn2, 1));
                                        else
                                            toggle1=isequal(unique([sweep.qubitnum]),circn1);
                                            toggle2=isequal(unique([sweep.qubitnum]),circn2);
                                        end
                                        if iter==1 || toggle1 %if so, recalculate expectation values
                                            %on the new eigenstates
                                            v1=Evect{1,circn1}; %get the unperturbed eigenvectors
                                            inductn1=cellfun(@(x) isequal(x,dict(Lmatrow).inductor),{obj.circuits(circn1).inductors.names});
                                            positions1=obj.circuits(dict(Lmatrow).circuit).inductors(inductn1).positions;
                                            if positions1(1)==positions1(2)
                                                Pb1=P{circn1}(:,:,positions1(1));
                                            else
                                                Pb1=P{circn1}(:,:,positions1(2))-P{circn1}(:,:,positions1(1));
                                            end
                                            H1(intindex)={ncon({conj(v1),Pb1,v1},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                        end
                                        if iter==1 || toggle2
                                            v2=Evect{1,circn2}; %get the unperturbed eigenvectors
                                            inductn2=cellfun(@(x) isequal(x,dict(Lmatcol).inductor),{obj.circuits(circn2).inductors.names});
                                            positions2=obj.circuits(dict(Lmatcol).circuit).inductors(inductn2).positions;
                                            if positions2(1)==positions2(2)
                                                Pb2=P{circn2}(:,:,positions2(1));
                                            else
                                                Pb2=P{circn2}(:,:,positions2(2))-P{circn2}(:,:,positions2(1));
                                            end
                                            H2(intindex)={ncon({conj(v2),Pb2,v2},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                        end
                                        Hintern=Id;
                                        if circn1==circn2
                                            Hintern{1,circn1}=H1{intindex}*H2{intindex}; %mediated self-interaction term
                                        else
                                            Hintern{1,circn1}=H1{intindex};
                                            Hintern{1,circn2}=H2{intindex};
                                        end
                                        if ~isempty(obj.lev2truncation)
                                            Hinttemp=obj.lev2truncate(Hintern,U);
                                            Hint(:,:,intindex)=Lmatg(Lmatrow,Lmatcol).*Hinttemp./obj.h*1e-9;
                                            intindex=intindex+1;
                                        else
                                            %every interachtion term is of the form
                                            %(L^-1)_ij*Phi_i*Phi_j with i!=j
                                            Hint(:,:,intindex)=Lmatg(Lmatrow,Lmatcol).*superkron(Hintern)./obj.h*1e-9;
                                            intindex=intindex+1;
                                        end
                                       % end
                                    end
                                end
                                if ~isempty(Cmatg) && ~isempty(find(Cmatg,1))
                                    for ind=find(triu(Cmatg,1))'
                                        [node1,node2]=ind2sub([length(Cmatg),length(Cmatg)],ind);
                                        circuitmarker=cumsum(sizes); %cumulative sum of the sequence of sizes of the capacitance
                                        %matrices
                                        circn1=find(node1<=circuitmarker,1);
                                        circn2=find(node2<=circuitmarker,1);
                                        if circn1>1
                                            node1=node1-(circuitmarker(circn1-1));
                                        end
                                        if circn2>1
                                            node2=node2-(circuitmarker(circn2-1));
                                        end
                                        if length(unique([sweep.qubitnum]))>1
                                            toggle1=~isempty(find([sweep.qubitnum]==circn1,1));
                                            toggle2=~isempty(find([sweep.qubitnum]==circn2,1));
                                        else
                                            toggle1=isequal(unique([sweep.qubitnum]),circn1);
                                            toggle2=isequal(unique([sweep.qubitnum]),circn2);
                                        end
                                        if iter==1 || toggle1
                                            v1=Evect{1,circn1}; %get the unperturbed eigenvectors
                                            Q1=Q{circn1}(:,:,node1);
                                            H1(intindex)={ncon({conj(v1),Q1,v1},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                        end
                                        if iter==1 || toggle2
                                            v2=Evect{1,circn2}; %get the unperturbed eigenvectors
                                            Q2=Q{circn2}(:,:,node2);
                                            H2(intindex)={ncon({conj(v2),Q2,v2},{[1 -1],[1 2],[2,-2]})}; %Hint=H1*H2
                                        end
                                        Hintern=Id;
                                        if circn1==circn2
                                            Hintern{1,circn1}=H1{intindex}*H2{intindex}; %mediated self-interaction term
                                        else
                                            Hintern{1,circn1}=H1{intindex};
                                            Hintern{1,circn2}=H2{intindex};
                                        end
                                        if ~isempty(obj.lev2truncation)
                                            Hinttemp=obj.lev2truncate(Hintern,U);
                                            Hint(:,:,intindex)=Cmatg(ind).*Hinttemp./obj.h*1e-9;
                                            intindex=intindex+1;
                                        else
                                            %every interachtion term is of the form
                                            %(C^-1)_ij*Q_i*Q_j with i!=j
                                            Hint(:,:,intindex)=Cmatg(ind)*superkron(Hintern)./obj.h*1e-9;
                                            intindex=intindex+1;
                                        end
                                    end
                                end
                                %sum unperturbed and interaction
                                %Hamiltonians
                                Htot=sum(Hbig,3)+sum(Hint,3).';
                                opts.maxit=500;
                                [evect,evals]=eigs(sparse(Htot),max(cat(2,initstate,finstate)),'SA',opts);
                                [~,order]=sort(real(diag(evals)));
                                if length(initstate)==1 && length(finstate)==1
                                    initvec=obj.normeigenvectors(evect(:,order(initstate)));
                                    finvec=obj.normeigenvectors(evect(:,order(finstate)));
                                    if initstate==finstate
                                        out(iter)=real(finvec'*operator*initvec*1e9);
                                    else
                                        out(iter)=finvec'*operator*initvec*1e9;
                                    end
                                    if isequal(iopname(1),'Q')
                                        out(1,iter)=out(1,iter)/obj.circuits(circuitnum).parammat.Cmat(str2double(iopname(2)),str2double(iopname(2)));
                                    end
                                elseif length(initstate)>1 && length(finstate)==1
                                    finvec=obj.normeigenvectors(evect(:,order(finstate)));
                                    initvec=evect(:,order(initstate));
                                    for inits=1:length(initstate)
                                        invs=obj.normeigenvectors(initvec(:,inits));
                                        out(inits,iter)=finvec'*operator*invs*1e9;
                                    end
                                    if isequal(iopname(1),'Q')
                                        out(:,iter)=out(:,iter)/obj.circuits(circuitnum).parammat.Cmat(str2double(iopname(2)),str2double(iopname(2)));
                                    end
                                elseif length(finstate)>1 && length(initstate)==1
                                    initvec=obj.normeigenvectors(evect(:,order(initstate)));
                                    finvec=evect(:,order(finstate));
                                    for finits=1:length(finstate)
                                        finv=obj.normeigenvectors(finvec(:,finits));
                                        out(finits,iter)=finv'*operator*initvec*1e9;
                                    end
                                    if isequal(iopname(1),'Q')
                                        out(:,iter)=out(:,iter)/obj.circuits(circuitnum).parammat.Cmat(str2double(iopname(2)),str2double(iopname(2)));
                                    end
                                elseif length(initstate)==length(finstate) && length(initstate)>1
                                    initvec=evect(:,order(initstate));
                                    finvec=evect(:,order(finstate));
                                    for inits=1:length(initstate)
                                        invs=obj.normeigenvectors(initvec(:,inits));
                                        finv=obj.normeigenvectors(finvec(:,inits));
                                        out(inits,iter)=finv'*operator*invs*1e9;
                                    end
                                    if isequal(iopname(1),'Q')
                                        out(:,iter)=out(:,iter)/obj.circuits(circuitnum).parammat.Cmat(str2double(iopname(2)),str2double(iopname(2)));
                                    end
                                end
                            end
                            obj.resetloadparammats; %restore parammats
                        end
                        close(f);
                        if nargin>6 && isequal(plotopt,'Plot') && optionp
                            figure
                            if length(initstate)==1 && length(finstate)==1
                                if initstate~=finstate
                                    plot(sweep(1).range,round(abs(out),5),'LineWidth',2);
                                    abssing='|';
                                else
                                    plot(sweep(1).range,round(real(out),5),'LineWidth',2);
                                    abssing='';
                                end
                                ylab=strcat('$',abssing,'\langle',num2str(initstate),'|');
                                if isequal(iopname,'Iz')
                                    ylab=strcat(ylab,'I_{z',num2str(circuitnum),'}|',num2str(finstate),'\rangle',abssing,'$',' (nA)');
                                elseif isequal(iopname,'Ix')
                                    ylab=strcat(ylab,'I_{x',num2str(circuitnum),'}|',num2str(finstate),'\rangle',abssing,'$',' (nA)');
                                elseif isequal(iopname(1),'Q')
                                    ylab=strcat(ylab,iopname,'_',num2str(circuitnum),'/C','|',num2str(finstate),'\rangle',abssing,'$',' (nV)');
                                end
                            elseif length(initstate)>1 && length(finstate)==1
                                plot(sweep(1).range,round(real(out),5),'LineWidth',2);
                                abssing='';
                                ylab=strcat('$',abssing,'\langle i|');
                                if isequal(iopname,'Iz')
                                    ylab=strcat(ylab,'I_{z',num2str(circuitnum),'}|',num2str(finstate),'\rangle',abssing,'$',' (nA)');
                                elseif isequal(iopname,'Ix')
                                    ylab=strcat(ylab,'I_{x',num2str(circuitnum),'}|',num2str(finstate),'\rangle',abssing,'$',' (nA)');
                                elseif isequal(iopname(1),'Q')
                                    ylab=strcat(ylab,iopname,'_',num2str(circuitnum),'/C','|',num2str(finstate),'\rangle',abssing,'$',' (nV)');
                                end
                                lab=cellfun(@(x) strcat('i=',x),sprintfc('%d',initstate),'UniformOutput',false);
                                legend(lab)
                            elseif length(finstate)>1 && length(initstate)==1
                                abssing='';
                                plot(sweep(1).range,round(real(out),5),'LineWidth',2);
                                ylab=strcat('$',abssing,'\langle',num2str(initstate),'|');
                                if isequal(iopname,'Iz')
                                    ylab=strcat(ylab,'I_{z',num2str(circuitnum),'}|f\rangle',abssing,'$',' (nA)');
                                elseif isequal(iopname,'Ix')
                                    ylab=strcat(ylab,'I_{x',num2str(circuitnum),'}|f\rangle',abssing,'$',' (nA)');
                                elseif isequal(iopname(1),'Q')
                                    ylab=strcat(ylab,iopname,'_',num2str(circuitnum),'/C','|f\rangle',abssing,'$',' (nV)');
                                end
                                lab=cellfun(@(x) strcat('f=',x),sprintfc('%d',initstate),'UniformOutput',false);
                                legend(lab)
                            elseif length(initstate)==length(finstate) && length(initstate)>1
                                abssing='';
                                plot(sweep(1).range,round(real(out),5),'LineWidth',2);
                                ylab=strcat('$',abssing,'\langle i|');
                                if isequal(iopname,'Iz')
                                    ylab=strcat(ylab,'I_{z',num2str(circuitnum),'}|f\rangle',abssing,'$',' (nA)');
                                elseif isequal(iopname,'Ix')
                                    ylab=strcat(ylab,'I_{x',num2str(circuitnum),'}|f\rangle',abssing,'$',' (nA)');
                                elseif isequal(iopname(1),'Q')
                                    ylab=strcat(ylab,iopname,'_',num2str(circuitnum),'/C','|f\rangle',abssing,'$',' (nV)');
                                end
                                if isequal(initstate,finstate)
                                    lab=cellfun(@(x) strcat('f=i=',x),sprintfc('%d',initstate),'UniformOutput',false);
                                    legend(lab)
                                end
                            end
                            xlim([min(sweep(1).range),max(sweep(1).range)]);
                            set(gca,'fontsize',18);
                            labelx=strcat('\textit{',sweep(1).parameter,'}${}_',num2str(sweep(1).qubitnum),'$');
                            xlabel(labelx,'FontSize',22,'Interpreter','latex')
                            ylabel(ylab,'FontSize',22,'Interpreter','latex')
                        end
                    else
                        fprintf("Inconsistent params ranges\n");
                    end
                end
            end
        end
        
        %% Further hierarchy for diagonalisation
        % finds the rotation mapping to the trucated basis of each subsystem
        % given the unperturbed hamiltonians
        function U=lev2rotations(obj,H0)
            numqubits=length(obj.circuits);
            if isempty(obj.lev2truncation)
                fprintf("2nd order truncation levels empty.\n")
            else
                subsnum=size(obj.lev2truncation,1); %number of subsystems
                U=cell(1,subsnum); %one rotation for every syubsystem
                for subind=1:subsnum
                    if subind==1
                        circs=cell2mat(obj.lev2truncation{subind,:}{1});
                    else
                        circs=cat(2,circs,cell2mat(obj.lev2truncation{subind,:}{1}));
                    end
                end
                if isequal(unique(circs),sort(circs)) %check that each circuit appears only once
                    if isequal(sort(circs),1:numqubits) %check that all circuits appear
                        H02=cell(1,subsnum);
                        for subind=1:subsnum
                            circs=cell2mat(obj.lev2truncation{subind,:}{1}); %circuits in subsystem subind
                            subsize=length(circs); %number of circuits in the subsystem
                            truncs=obj.lev1truncation(circs); %level 1 truncations
                            Id=cellfun(@eye,num2cell(truncs),'UniformOutput',0);
                            H0new=zeros(prod(truncs));
                            for circn=1:subsize
                                Hn=Id;
                                Hn(1,circn)=H0(1,circs(circn));
                                H0new=H0new+superkron(Hn);
                            end
                            [U{1,subind},H02{subind}]=eigs(H0new,obj.lev2truncation{subind,:}{2},'sr');
                            %U{1,subind} contains the first
                            %obj.lev2truncation{subind,:}{2} eigenvectors
                            %of the sum of the unperturbed Hamiltonians of
                            %the circuits in the subsystem, this allows us
                            %to project the interaction terms in the
                            %hamiltonian onto these states
                        end
                        if ~isempty(obj.lev3truncation)
                            subsnum=size(obj.lev3truncation,1); %number of upper subsystems
                            U2=cell(1,subsnum); %one rotation for every syubsystem
                            for subind=1:subsnum
                                if subind==1
                                    circs=cell2mat(obj.lev3truncation{subind,:}{1});
                                else
                                    circs=cat(2,circs,cell2mat(obj.lev3truncation{subind,:}{1}));
                                end
                            end
                            if isequal(unique(circs),sort(circs)) %check that each lowe subsystem appears only once
                                if isequal(sort(circs),1:length(obj.lev2truncation)) %check that all circuits appear
                                    truncs=(cellfun(@(x) x{2},obj.lev2truncation))';
                                    for subind=1:subsnum
                                        circs=cell2mat(obj.lev3truncation{subind,:}{1}); %circuits in subsystem subind
                                        subsize=length(circs); %number of circuits in the subsystem
                                        truncsi=truncs(circs); %level 2 truncations
                                        Id=cellfun(@eye,num2cell(truncsi),'UniformOutput',0);
                                        H0new=zeros(prod(truncsi));
                                        for circn=1:subsize
                                            Hn=Id;
                                            Hn(1,circn)=H02(1,circs(circn));
                                            H0new=H0new+superkron(Hn);
                                        end
                                        [U2{1,subind},~]=eigs(H0new,obj.lev3truncation{subind,:}{2},'sr');
                                        %U{1,subind} contains the first
                                        %obj.lev2truncation{subind,:}{2} eigenvectors
                                        %of the sum of the unperturbed Hamiltonians of
                                        %the circuits in the subsystem, this allows us
                                        %to project the interaction terms in the
                                        %hamiltonian onto these states
                                    end
                                end
                            end
                        end
                        if ~isempty(obj.lev3truncation)
                            U=cat(2,U,U2);
                        end
                    else
                        fprintf("Check that all circuits appear in exactly 1 subsystem\n")
                    end
                else
                    fprintf("Some circuits appear in more than 1 subsystem.\n")
                end
            end
        end
        
        %% Build 2nd level h.d. Hamiltonian terms.
        % Given an Hamiltonian term and the corresponding operators acting
        % on each circuit in the rapresentation defined in the level 1 of
        % truncation, this function returns the expression of the
        % Hamiltonian term in the 2nd order truncation basis
        function Hnew=lev2truncate(obj,H,U)
            numqubits=length(obj.circuits); %number or circuits in the system
            if length(H)==numqubits %check that operators are given for every circuit
                Hnewtemp=cell(1,length(U)); %one term for every subsystem
                for subind=1:length(obj.lev2truncation) %for every lower subsystem
                    circs=cell2mat(obj.lev2truncation{subind,:}{1}); %circuits in subsystem subind
                    subsize=length(circs); %number of circuits in the subsystem
                    truncs=obj.lev1truncation(circs); %corresponding truncations
                    Id=cellfun(@eye,num2cell(truncs),'UniformOutput',0);
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
        %% Define pairwise interaction parameters
        function setinteractions(obj,circuit1,circuit2,couplingtype,coupledelem1,coupledelem2,strength)
            % SETINTERACTIONS Defines the interaction parameters
            % 
            % obj.setinteractions(circuit1,circuit2,couplingtype,coupledelem1,coupledelem2,strength)
            % defines an interaction between "circuit1" and "circuit2" (integer
            % numbers) of type couplingtype ('Ind' or 'Cap') involving the
            % two branch inductors/circuit nodes "coupledelem1" and 
            % "coupledelem2" and with mutual inductance/coupling
            % capacitance "strength"
            if nargin<6
                fprintf('Please, specify the correct inputs\n');
            elseif circuit1>length(obj.circuitnames) || circuit2>length(obj.circuitnames)
                fprintf('Please, specify two correct circuit numbers\n');
            else
                if isempty(obj.interactions)
                    obj.interactions=struct('interpair',{{circuit1,circuit2}});
                    obj.interactions.ctype=couplingtype;
                    if isequal(couplingtype,'Ind')
                        obj.interactions.inductors={coupledelem1,coupledelem2};
                        obj.interactions.M=strength;
                        obj.interactions.C=0;
                        obj.interactions.couplednodes={};
                    elseif isequal(couplingtype,'Cap')
                        obj.interactions.inductors={};
                        obj.interactions.M=0;
                        obj.interactions.C=strength;
                        obj.interactions.couplednodes={coupledelem1,coupledelem2};
                    else
                        fprintf('Specify a correct interaction type\n');
                    end
                else
                    obj.interactions(end+1).interpair={circuit1,circuit2};
                    obj.interactions(end).ctype=couplingtype;
                    if isequal(couplingtype,'Ind')
                        obj.interactions(end).inductors={coupledelem1,coupledelem2};
                        obj.interactions(end).M=strength;
                        obj.interactions(end).C=0;
                        obj.interactions(end).couplednodes={};
                    elseif isequal(couplingtype,'Cap')
                        obj.interactions(end).inductors={};
                        obj.interactions(end).M=0;
                        obj.interactions(end).C=strength;
                        obj.interactions(end).couplednodes={coupledelem1,coupledelem2};
                    else
                        fprintf('Specify a correct interaction type\n');
                    end
                end
            end
        end
        
        %% Build the mutual inductance matrix for the coupled branches
        % and a dictionary containing the reference to the inductors for
        % every position in the matrix
        function [dict,Lmat]=buildinductancemat(obj)
            dict=struct();%initialise the empty dictionary
            %numinter=length(obj.interactions);
            index=1;
            %for intnum=1:numinter
            for intnum=find(cellfun(@(x) isequal(x,'Ind'),{obj.interactions.ctype}))
                %if isequal(obj.interactions(intnum).ctype,'Ind')
                    for pairind=1:2
                        if index==1
                            dict=struct('index',1,'circuit',obj.interactions(intnum).interpair{1},'inductor',obj.interactions(intnum).inductors{1});
                            index=2;
                        else %check that every inductor only appears once before updating the dictionary
                            if ~(any([dict.circuit]==obj.interactions(intnum).interpair{pairind} & cellfun(@(x) isequal(x,obj.interactions(intnum).inductors{pairind}),{dict.inductor})))
                            %if ~(any([dict.circuit]==obj.interactions(intnum).interpair{pairind}) && any(cellfun(@(x) isequal(x,obj.interactions(intnum).inductors{pairind}),{dict.inductor})))
                                dict(end+1)=struct('index',index,'circuit',obj.interactions(intnum).interpair{pairind},'inductor',obj.interactions(intnum).inductors{pairind});
                                index=index+1;
                            end
                        end
                    end
               % end
            end %build the global inductance matrix
            Lmat=zeros(length(dict)); %build a mutual inductance matrix with
            % size equal to the number of interacting branches
            for i=1:length(dict)
                for j=i:length(dict)
                    if j==i
                        Lmat(i,j)=obj.circuits(dict(i).circuit).parameters.(dict(i).inductor);
                    else
                        if isempty(find(cellfun(@(x) isequal(x,{dict(i).circuit,dict(j).circuit}),{obj.interactions.interpair}), 1))
                            if isempty(find(cellfun(@(x) isequal(x,{dict(j).circuit,dict(i).circuit}),{obj.interactions.interpair}), 1))
                                Lmat(i,j)=0; %if the two branches are not interacting, then L_ij=0
                            else
                                for inters=1:length(find(cellfun(@(x) isequal(x,{dict(j).circuit,dict(i).circuit}),{obj.interactions.interpair})))
                                    ind=find(cellfun(@(x) isequal(x,{dict(j).circuit,dict(i).circuit}),{obj.interactions.interpair}));
                                    ind=ind(inters);
                                    %Check that there is an inductive interaction
                                    %between the two circuits and that the 
                                    if isequal(obj.interactions(ind).ctype,'Ind') && strcmp(obj.interactions(ind).inductors{1},dict(j).inductor) && strcmp(obj.interactions(ind).inductors{2},dict(i).inductor)
                                        Lmat(i,j)=-obj.interactions(ind).M;
                                        %L_ij=-M=-mutual inductive
                                        %coefficient otherwise
                                    end
                                end
                            end
                        else
                            for inters=1:length(find(cellfun(@(x) isequal(x,{dict(i).circuit,dict(j).circuit}),{obj.interactions.interpair})))
                                ind=find(cellfun(@(x) isequal(x,{dict(i).circuit,dict(j).circuit}),{obj.interactions.interpair}));
                                ind=ind(inters);
                                if isequal(obj.interactions(ind).ctype,'Ind') && strcmp(obj.interactions(ind).inductors{1},dict(i).inductor) && strcmp(obj.interactions(ind).inductors{2},dict(j).inductor)
                                    Lmat(i,j)=-obj.interactions(ind).M;
                                end
                            end
                        end
                    end
                end
            end
            Lmat=Lmat+triu(Lmat,1)'; %fill lower triangle
        end
        
        %% Build global capacitance matrix
        function [Cmat,Cmatint,sizes]=buildcapacitancemat(obj)
            Cmat=cell(1,length(obj.circuits));
            for circ=1:length(obj.circuits)
                Cmat{1,circ}=obj.circuits(circ).parammat.Cmat;%+obj.circuits(circ).gatescap
            end
            sizes=cellfun(@(x) length(x), Cmat);
            Cmat=blkdiag(Cmat{:});
            for ind=find(cellfun(@(x) isequal(x,'Cap'),{obj.interactions.ctype}))
                circ1=obj.interactions(ind).interpair{1};
                node1=obj.interactions(ind).couplednodes{1}+sum(sizes(1:circ1-1));
                circ2=obj.interactions(ind).interpair{2};
                node2=obj.interactions(ind).couplednodes{2}+sum(sizes(1:circ2-1));
                Cmat(node1,node2)=Cmat(node1,node2)-obj.interactions(ind).C;
                Cmat(node2,node1)=Cmat(node2,node1)-obj.interactions(ind).C;
                Cmat(node1,node1)=Cmat(node1,node1)+obj.interactions(ind).C;
                Cmat(node2,node2)=Cmat(node2,node2)+obj.interactions(ind).C;
            end
            Cmatint=inv(Cmat);
            Cmat=zeros(length(Cmatint));
            for circ=1:length(obj.circuits)
                Cmat(1+sum(sizes(1:circ-1)):sum(sizes(1:circ)),1+sum(sizes(1:circ-1)):sum(sizes(1:circ)))=Cmatint(1+sum(sizes(1:circ-1)):sum(sizes(1:circ)),1+sum(sizes(1:circ-1)):sum(sizes(1:circ)));
                Cmatint(1+sum(sizes(1:circ-1)):sum(sizes(1:circ)),1+sum(sizes(1:circ-1)):sum(sizes(1:circ)))=0;
            end
            Cmat=inv(Cmat);
        end
        
        %% Changes Lmat and Cmat to consider the loading (coupling) inductance/capacitance
        function setloadparammats(obj)
            numqubits=length(obj.circuits);
            Cmat=cell(1,numqubits);
            Lmat=cell(1,numqubits);
            for circuit=1:numqubits
                if ~isempty(obj.circuits(circuit).parammatext)
                    obj.circuits(circuit).parammatext={}; %make sure that the loaded matrices have not been created already
                end
                Cmat{1,circuit}=obj.circuits(circuit).parammat.Cmat;%+obj.circuits(circuit).gatescap;
                Lmat{1,circuit}=obj.circuits(circuit).parammat.Lmat;
            end
            sizes=cellfun(@(x) length(x), Cmat);
            if any(cellfun(@(x) isequal(x,'Cap'),{obj.interactions.ctype}))
                [Cmatg,~,~]=obj.buildcapacitancemat();
                for circuit=1:numqubits
                    Cmat{1,circuit}=Cmatg(1+sum(sizes(1:circuit-1)):sum(sizes(1:circuit)),1+sum(sizes(1:circuit-1)):sum(sizes(1:circuit)));
                end
            end
            if any(cellfun(@(x) isequal(x,'Ind'),{obj.interactions.ctype}))
                [dict,Lmatg]=obj.buildinductancemat(); %get the global branch indcutance matrix
                Lmatg=inv(Lmatg); %take the inverse
                for ind=1:length(dict) %for every interacting branch
                    circn=dict(ind).circuit; %corresponding circuit
                    inductn=cellfun(@(x) isequal(x,dict(ind).inductor),{obj.circuits(circn).inductors.names}); %circuit inductor
                    position=obj.circuits(circn).inductors(inductn).positions; %position of the inductor in the circuit
                    if position(1)==position(2) %rescale Lmat elements
                        Lmat{1,circn}(position(1),position(1))=Lmat{1,circn}(position(1),position(1))-1/obj.circuits(circn).parameters.(dict(ind).inductor)+Lmatg(ind,ind);
                    else
                        Lmat{1,circn}(position(1),position(1))=Lmat{1,circn}(position(1),position(1))-1/obj.circuits(circn).parameters.(dict(ind).inductor)+Lmatg(ind,ind);
                        Lmat{1,circn}(position(2),position(2))=Lmat{1,circn}(position(2),position(2))-1/obj.circuits(circn).parameters.(dict(ind).inductor)+Lmatg(ind,ind);
                        Lmat{1,circn}(position(1),position(2))=Lmat{1,circn}(position(1),position(2))+1/obj.circuits(circn).parameters.(dict(ind).inductor)-Lmatg(ind,ind);
                        Lmat{1,circn}(position(2),position(1))=Lmat{1,circn}(position(2),position(1))+1/obj.circuits(circn).parameters.(dict(ind).inductor)-Lmatg(ind,ind);
                    end
                end
            end
            for circuit=1:numqubits
                obj.circuits(circuit).parammatext=struct('Cmat',Cmat{1,circuit},'Lmat',Lmat{1,circuit},...
                            'Ejmat',obj.circuits(circuit).parammat.Ejmat);
            end
        end
        
        %% Resets the original Lmat and Cmat
        function resetloadparammats(obj)
            numqubits=length(obj.circuitnames);
            for circuit=1:numqubits
                obj.circuits(circuit).parammatext={};
            end
        end
        
        %% Set first value of the eigenvector to be real
        function out=normeigenvectors(~,ev)
            out=zeros(size(ev));
            for column=1:size(ev,2)
                evtemp=ev(:,column);
                evtemp=evtemp./evtemp(find(abs(evtemp)>1e-7,1))*abs(evtemp(find(abs(evtemp)>1e-7,1))); %Divide by the first vector element significantly differrent from 0
                out(:,column)=evtemp;
            end
        end
        
        %% Changes one parameter of one of the qubits
        function obj=changeparam(obj,qubitn,paramname,value)
            if qubitn>length(obj.circuitnames)
                fprintf("Invalid qubit number. There are only %i qubits in the system\n",length(obj.circuitnames));
            end
            obj.circuits(qubitn).parameters.(paramname)=value;
        end
        
        %% Changes one flux bias of one of the qubits
        function obj=changebias(obj,qubitn,biasname,value)
            if qubitn>length(obj.circuitnames)
                fprintf("Invalid qubit number. There are only %i qubits in the system\n",length(obj.circuitnames));
            end
            obj.circuits(qubitn).fluxb.(biasname)=value;
        end
        
        %% Prints the parameters of one qubit
        function out=showparams(obj,qubitn)
            if qubitn>length(obj.circuitnames)
                fprintf("Invalid qubit number. There are only %i qubits in the system\n",length(obj.circuitnames));
            end
            out=obj.circuits(qubitn).parameters;
        end
        
        %% Prints the flux biases of one qubit
        function out=showbias(obj,qubitn)
            if qubitn>length(obj.circuitnames)
                fprintf("Invalid qubit number. There are only %i qubits in the system\n",length(obj.circuitnames));
            end
            out=obj.circuits(qubitn).fluxb;
        end
        
        %% Prints the operators that can be used for coupling
        function out=showops(obj,qubitn)
            if qubitn>length(obj.circuitnames)
                fprintf("Invalid qubit number. There are only %i qubits in the system\n",length(obj.circuitnames));
            end
            out=obj.circuits(qubitn).oplist;
        end
        
        %% Paramsweep method: defines a parameter sweep
        function paramsweep(obj,sweepname,qubitn,param,range)
            if iscell(param)
                toggle=0;
                for paramn=1:length(param)
                    if isfield(obj.circuits(qubitn(paramn)).parameters,param{paramn}) && isempty(obj.sweeps)
                        obj.sweeps=struct('sweepname',sweepname,'qubitnum',qubitn(paramn),'sweeptype','param','parameter',param{paramn},'range',range{paramn});
                    elseif isfield(obj.circuits(qubitn(paramn)).parameters,param{paramn})
                        if ~sum(cellfun(@(x,y) isequal(x,y),{obj.sweeps.sweepname},repmat({sweepname},1,length(obj.sweeps)))) || toggle
                            obj.sweeps(end+1)=struct('sweepname',sweepname,'qubitnum',qubitn(paramn),'sweeptype','param','parameter',param{paramn},'range',range{paramn});
                        else
                            fprintf('Sweep name already in use\n');
                        end
                    else
                        fprintf('Use a valid param sweep parameter\n');
                    end
                    toggle=toggle+1;
                end
            else
                if isfield(obj.circuits(qubitn).parameters,param) && isempty(obj.sweeps)
                    obj.sweeps=struct('sweepname',sweepname,'qubitnum',qubitn,'sweeptype','param','parameter',param,'range',range);
                elseif isfield(obj.circuits(qubitn).parameters,param)
                    if ~sum(cellfun(@(x,y) isequal(x,y),{obj.sweeps.sweepname},repmat({sweepname},1,length(obj.sweeps))))
                        obj.sweeps(end+1)=struct('sweepname',sweepname,'qubitnum',qubitn,'sweeptype','param','parameter',param,'range',range);
                    else
                        fprintf('Sweep name already in use\n');
                    end
                else
                    fprintf('Use a valid param sweep parameter\n');
                end
            end
        end
        
        %% Flux sweep method: defines a flux bias sweep
        function fluxsweep(obj,sweepname,qubitn,bias,range)
            if any(arrayfun(@(x) isempty(obj.circuits(x).closures),qubitn))
                fprintf("This function is only available for circuits with at least one irreducible loop.\n")
            else
                if iscell(bias)
                    toggle=0;
                    for biasn=1:length(bias)
                        if isfield(obj.circuits(qubitn(biasn)).fluxb,bias{biasn}) && isempty(obj.sweeps)
                            obj.sweeps=struct('sweepname',sweepname,'qubitnum',qubitn(biasn),'sweeptype','flux','parameter',bias{biasn},'range',range{biasn});
                        elseif isfield(obj.circuits(qubitn(biasn)).fluxb,bias{biasn})
                            if ~sum(cellfun(@(x,y) isequal(x,y),{obj.sweeps.sweepname},repmat({sweepname},1,length(obj.sweeps)))) || toggle
                                obj.sweeps(end+1)=struct('sweepname',sweepname,'qubitnum',qubitn(biasn),'sweeptype','flux','parameter',bias{biasn},'range',range{biasn});
                            else
                                fprintf('Sweep name already in use\n');
                            end
                        else
                            fprintf('Use a valid flux sweep parameter\n');
                        end
                        toggle=toggle+1;
                    end
                else
                    if isfield(obj.circuits(qubitn).fluxb,bias) && isempty(obj.sweeps)
                        obj.sweeps=struct('sweepname',sweepname,'qubitnum',qubitn,'sweeptype','flux','parameter',bias,'range',range);
                    elseif isfield(obj.circuits(qubitn).fluxb,bias)
                        if ~sum(cellfun(@(x,y) isequal(x,y),{obj.sweeps.sweepname},repmat({sweepname},1,length(obj.sweeps))))
                            obj.sweeps(end+1)=struct('sweepname',sweepname,'qubitnum',qubitn,'sweeptype','flux','parameter',bias,'range',range);
                        else
                            fprintf('Sweep name already in use\n');
                        end
                    else
                        fprintf('Use a valid flux sweep parameter\n');
                    end
                end
            end
        end   
    end
end