%Flux qubits
qubit01=FluxQubit("fluxRF");
qubit02=FluxQubit("fluxRF");
%Capacitively shunted variant
qubit01.parameters.Lq=1.6e-9;
qubit02.parameters.Lq=1.6e-9;
qubit01.parameters.Csh=4.5e-14;
qubit02.parameters.Csh=4.5e-14;
%Unshunted variant
qubit01.parameters.Lq=3.2e-9;
qubit02.parameters.Lq=3.2e-9;
qubit01.parameters.Csh=0;
qubit02.parameters.Csh=0;

%Averin coupler (Cooper pair box circuit)
%Using parameters from Louis's notebooks
coupler=FluxQubit("SCP");
coupler.parameters.Ic=4e-8;
coupler.parameters.Csh=1.8e-15+2.5e-15; %shunting capacitance across the junction
coupler.voltb.Cg=2.5e-15; %gate capacitor
coupler.voltb.Qg=0;
coupler.voltsweep("vsweep",1,linspace(-1,1,101))
%Coupler spectrum
coupler.sweepE("vsweep",10,'Relative',1,'Plot',1);
%Minimum gap of 16.9GHz (at Qg=0.5)

%Coupled circuit: the two qubits are capacitively coupled to the coupler
%circuit
coupled=QubitsInteract(qubit01,qubit02,coupler);
coupled.lev1truncation=[10,10 10];
coupled.setinteractions(1,3,'Cap',1,1,5e-15); %Coupling capacitors=5fF
coupled.setinteractions(2,3,'Cap',1,1,5e-15);

%Set static biases: qubits biased at the symmetry point
coupler.voltb.Qg=0.5;
coupled.changebias(1,'fz',0.5);
coupled.changebias(2,'fz',0.5);

%Pauli coefficient names
coefficients={["z","I","I"];["I","z","I"];["x","I","I"];["I","x","I"];...
    ["y","I","I"];["I","y","I"];["x","x","I"];["x","z","I"];["y","y","I"];...
    ["z","x","I"];["z","z","I"]};

%Coupler bias charge sweep (a voltage sweep method is not available atm)
Qc=linspace(0,2,101);
levelsout=[];
coeffout=[];
for i=1:length(Qc)
    fprintf("%i\n",i)
    coupler.voltb.Qg=Qc(i);
    levelsout=cat(1,levelsout,coupled.calculateE(40,'Relative',1)'); %energy levels
    coeffout=cat(1,coeffout,coupled.coeffP(coefficients,[1 2])); %Pauli coefficients
end

figure
h=plot(Qc,coeffout(:,1:6),'LineWidth',2);
cmap=hsv(6);
for i=1:6
set(h(i),'Color',cmap(i,:))
end
legend(h,{'$h_{zI}$','$h_{Iz}$','$h_{xI}$','$h_{Ix}$','$h_{yI}$','$h_{Iy}$'},'FontSize',20,'Interpreter','latex')
legend('boxoff')
set(gca,'fontsize',18);
xlabel('$Q_{g,c}/2e$','FontSize',22,'Interpreter','latex')
ylabel('Coefficients (GHz)','FontSize',22,'Interpreter','latex')

figure
h=plot(Qc,coeffout(:,7:11),'LineWidth',2);
cmap=hsv(4);
for i=1:4
set(h(i),'Color',cmap(i,:))
end
legend(h,{'$h_{xx}$','$h_{xz}$','$h_{yy}$','$h_{zx}$','$h_{zz}$'},'NumColumns',3,'FontSize',20,'Interpreter','latex')
set(gca,'fontsize',18);
xlabel('$Q_{g,c}/2e$','FontSize',22,'Interpreter','latex')
ylabel('Coefficients (GHz)','FontSize',22,'Interpreter','latex')
legend('boxoff')

%Verify that the coefficients produce the right spectrum:
%Pauli matrices:
sx=[[0,1];[1,0]];
sy=[[0,-1i];[1i,0]];
sz=[[1,0];[0,-1]];
sx1=kron(sx,eye(2));
sx2=kron(eye(2),sx);
sy1=kron(sy,eye(2));
sy2=kron(eye(2),sy);
sz1=kron(sz,eye(2));
sz2=kron(eye(2),sz);

levelsout2=zeros(length(Qc),4);
for i=1:length(Qc)
    H=coeffout(i,1)*sz1+coeffout(i,2)*sz2+coeffout(i,3)*sx1+coeffout(i,4)*sx2+...
        coeffout(i,5)*sy1+coeffout(i,6)*sy2+coeffout(i,7)*sx1*sx2+coeffout(i,8)*sx1*sz2+...
        coeffout(i,9)*sy1*sy2+coeffout(i,10)*sz1*sx2+coeffout(i,11)*sz1*sz2;
    en=eig(H);
    levelsout2(i,:)=en-en(1);
end

figure
plot(Qc,levelsout,'LineWidth',2);
hold
plot(Qc,levelsout2,'k--','LineWidth',2);
set(gca,'fontsize',18);
xlabel('$Q_{g,c}/2e$','FontSize',22,'Interpreter','latex')
ylabel('$E_i-E_0$ (GHz)','FontSize',22,'Interpreter','latex')
axis([xlim 0 15]);

%Coupling capacitors sweep:
coupler.voltb.Qg=0.5;
Cc=linspace(1e-16,4e-14,101);
levelsoutc=[];
coeffoutc=[];
for i=1:length(Qc)
    fprintf("%i\n",i)
    coupled.interactions(1).C=Cc(i);
    coupled.interactions(2).C=Cc(i);
    levelsoutc=cat(1,levelsoutc,coupled.calculateE(40,'Relative',1)');
    coeffoutc=cat(1,coeffoutc,coupled.coeffP(coefficients,[1 2]));
end

%Charge qubits, using Carten papers params:
%Parameters set 1:
%EJi=0.2EJ, EC/EJ=2
qubit01=FluxQubit("SCP");
qubit02=FluxQubit("SCP");
qubit01.parameters.Ic=0.8e-9;
qubit02.parameters.Ic=0.8e-9;
qubit01.parameters.Csh=9.3e-14;
qubit02.parameters.Csh=9.3e-14;
coupler=FluxQubit("SCP");
coupler.parameters.Ic=4e-9;
coupler.parameters.Csh=9e-15; %shunting capacitance across the junction
coupler.voltb.Cg=1e-15; %gate capacitor

%Parameters set 2:
%EJi=0.2EJ, EC/EJ=0.5
qubit01.parameters.Ic=1e-8;
qubit01.parameters.Csh=4.7e-15;
qubit02.parameters.Ic=1e-8;
qubit02.parameters.Csh=4.7e-15;
coupler=FluxQubit("SCP");
coupler.parameters.Ic=5e-8;
coupler.parameters.Csh=0;
coupler.voltb.Cg=0.9e-15;

coupled.setloadparammats; %Include capacitive loading before calculating the
%coupler and qubits unperturbed spectra
coupler.voltb.Qg=0;
coupler.voltsweep("vsweep",1,linspace(-1,1,101))
coupler.sweepE("vsweep",10,'Relative',1,'Plot',1);
qubit01.voltb.Qg=0;
qubit01.voltsweep("vsweep",1,linspace(-1,1,101))
qubit01.sweepE("vsweep",10,'Relative',1,'Plot',1);
coupled.resetloadparammats;

coupled=QubitsInteract(qubit01,qubit02,coupler);
coupled.lev1truncation=[10,10 10];
coupled.setinteractions(1,3,'Cap',1,1,5e-15);
coupled.setinteractions(2,3,'Cap',1,1,5e-15);

coupler.voltb.Qg=0.5;
qubit01.voltb.Qg=0.5;
qubit02.voltb.Qg=0.5;

coefficients={["z","I","I"];["I","z","I"];["x","I","I"];["I","x","I"];...
    ["y","I","I"];["I","y","I"];["x","x","I"];["x","z","I"];["y","y","I"];...
    ["z","x","I"];["z","z","I"]};

Qc=linspace(0,1,101);
levelsout=[];
coeffout=[];
for i=1:length(Qc)
    fprintf("%i\n",i)
    coupler.voltb.Qg=Qc(i);
    levelsout=cat(1,levelsout,coupled.calculateE(40,'Relative',1)');
    coeffout=cat(1,coeffout,coupled.coeffP(coefficients,[1 2]));
end

figure
h=plot(Qc,coeffout(:,1:6)*1e3,'LineWidth',2);
axis([xlim -300 100]);
set(gca,'fontsize',18);
xlabel('$Q_{g,c}/2e$','FontSize',22,'Interpreter','latex')
ylabel('Coefficients (MHz)','FontSize',22,'Interpreter','latex')
legend(h,{'$h_{zI}$','$h_{Iz}$','$h_{xI}$','$h_{Ix}$','$h_{yI}$','$h_{Iy}$'},'NumColumns',3,'FontSize',20,'Interpreter','latex')
legend('boxoff')

figure
h=plot(Qc,coeffout(:,7:11)*1e3,'LineWidth',2);
axis([xlim -10 10]);
set(gca,'fontsize',18);
xlabel('$Q_{g,c}/2e$','FontSize',22,'Interpreter','latex')
ylabel('Coefficients (MHz)','FontSize',22,'Interpreter','latex')
legend(h,{'$h_{xx}$','$h_{xz}$','$h_{yy}$','$h_{zx}$','$h_{zz}$'},'NumColumns',3,'FontSize',20,'Interpreter','latex')
legend('boxoff')

levelsout2=zeros(length(Qc),4);
for i=1:length(Qc)
    H=coeffout(i,1)*sz1+coeffout(i,2)*sz2+coeffout(i,3)*sx1+coeffout(i,4)*sx2+...
        coeffout(i,5)*sy1+coeffout(i,6)*sy2+coeffout(i,7)*sx1*sx2+coeffout(i,8)*sx1*sz2+...
        coeffout(i,9)*sy1*sy2+coeffout(i,10)*sz1*sx2+coeffout(i,11)*sz1*sz2;
    en=eig(H);
    levelsout2(i,:)=en-en(1);
end

%spectrum
figure
plot(Qc,levelsout,'LineWidth',2)
hold
plot(Qc,levelsout2,'k--','LineWidth',2)
axis([xlim 0 1]);
set(gca,'fontsize',18);
xlabel('$Q_{g,c}/2e$','FontSize',22,'Interpreter','latex')
ylabel('$E_i-E_0$ (GHz)','FontSize',22,'Interpreter','latex')

% coupled.setloadparammats;
% c=5e-15/qubit01.parammat.Cmat;
% Ec=2*e^2/coupler.parammat.Cmat/6.62607004e-34*1e-9;
% coupled.resetloadparammats;
% figure
% h=plot(Qc,4*coeffout(:,[11 9])/c^2/Ec,'LineWidth',2);
% axis([xlim -3.3 1.6]);
% set(gca,'fontsize',18);
% xlabel('$Q_{g,c}/2e$','FontSize',22,'Interpreter','latex')
% ylabel('$\lambda_{c,i}/c_1c_2E_C$','FontSize',22,'Interpreter','latex')
% legend({'$\lambda_c$','$\lambda_i$'},'FontSize',20,'Interpreter','latex')
% legend('boxoff')

