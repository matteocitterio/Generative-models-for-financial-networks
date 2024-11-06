def MontecarloPrice(self, t, T, n_samples = 1000):
         """
         Performs Montecarlo simulation of CIR, generating `n_samples` paths and then computes expectation out of it from 
         which we can get the bond price at time t with maturity at time T

         Parameters
         -----------
         - t : `int`
            Current time at which we want to know the bond price
         - T : `int`
            Maturity of the bond contract
         - n_samples : `int` (default : 10 000)
            Number of paths to generate for the computing the MC expectation
         Returns
         -------
         - P : `float`
            Price of the bond at time t with maturity T (p(t,T))
         """

         if t == T:
               # By definition, p(t,t) = 1 \forall t
               return 1

         # Monte Carlo simulations
         #PLEASE NOTE that in numpy.prod the right extreme is not included in the productory, therefore if we want times t<=T we need to take
         #up to T+1
         #PLEASE NOTE that, conversely, the left extreme is included. When computing B_t we want to include it as it cancels out in the ratio 
         # with t_0. However, in  the actual p(t_0, T) computation we dont want to include it, therefore we take t+1 such that we get t_0 < t <= T.
         MC_sims = get_CIR(self.time_grid[t:T+1],
                                 self.alpha,
                                 self.b,
                                 self.sigma,
                                 self.CIRProcess[t],
                                 n_samples = n_samples,
                                 seed = True,
                                 seed_number=0)
         
         prod = np.prod(1+MC_sims[1:]*(1./365), axis=1)
         return np.mean(prod**(-1))

    def SwapRate(self, t_0, T):
        """
        Computes OIS swap rate, the fixed rate that makes the contract value equals to 0 at time t_0

        Parameters
        -----------
        - t_0 : `int`
           Initial time of the contract
        - T : `int`
           Maturity
           
        Returns
        --------
        - K : `Float` 
           OIS swap rate
        """

        tau = np.array((T - t_0) / 365)

        condition = (tau <= 0)

        if self.seed:
            price = self.Price
      
        else:
            price = self.MontecarloPrice
        
        return np.where(condition, 0, (tau>0)*(-1 / tau) * ((price(t_0, T) - 1) / price(t_0, T)) )

def SimulateEdge(self, i, j, contracts, n_active_contr, n_active_days, steps_ahead = 1):
        """
        It simulates the contract arrival process on the edge (i,j)

        Parameters
        -----------
        - i : `int`
           One of the node in the edge (i,j)
        - j : `int`
           The other node in the edge (i,j)
           DA FAREEEEE
        """

        # Check i!=j as an institution cant trade with itself
        assert (i!=j)
 
        #PER IL MOMENTO NON CONSIDERO SOLO GIORNI ATTIVI, PRENDO TUTTI I GIORNI E VEDIAMO
        
        #Retrieve the number of feautures per contract
        n_contract_features = len(contracts[0].get_contract_features(contracts[0].t_0))

        #Name the CIR process
        CIRProcess = torch.tensor(self.CIRProcess.reshape(-1,1))

        #I will fill this one with the indexis of the active contracts
        #This list will contain only active contracts
        X = []
        y_margin = []
        conditioning = []
        active_days = []
        
        days_with_active_contracts = 0
        for t in range(self.TotPoints):
            n_active_contracts = np.sum([contract.is_active(t) for contract in contracts])
            if n_active_contracts > 0:
               days_with_active_contracts +=1

        y_benchmark = torch.zeros((days_with_active_contracts, steps_ahead))

        h=0

        loop = tqdm(range(self.TotPoints - 1 - steps_ahead + 1), desc=f'{i,j}')

        for i_time, t in enumerate(loop):
            
            active_contracts = [contract for contract in contracts if contract.is_active(t)]
        
            #This will contain the contracts at a certain time
            temp_contract_row = torch.zeros((n_active_contr*n_contract_features))
            temp_y_margin = 0

            for i_contract, contract in enumerate(active_contracts):
                  
                  #Retrieve contract features
                  contract_features = contract.get_contract_features(t)    
                  #Fill the row    
                  temp_contract_row[i_contract*len(contract_features) : (i_contract+1)*len(contract_features)] = contract_features
                  #Fill the target accordingly
                  temp_y_margin += contract.GetVariationMargin(t)
                  
            #Append the computed quantities if contract array is different from just zeros
            if not torch.all(torch.eq(temp_contract_row, torch.zeros_like(temp_contract_row))):
                  
                  active_days.append(i_time)
                  
                  X.append(temp_contract_row)
                  y_margin.append(temp_y_margin)

                  conditioning.append(CIRProcess[i_time])

                  for eta in range(steps_ahead):
                  
                     y_benchmark[h, eta] = self.ProvideBenchmark(t_l=t, steps_ahead=eta+1, contracts=contracts, n_simulations=100)
            
                  h+=1

        #Transform quantities in tensors
        X = torch.vstack(X)
        active_days = torch.tensor(active_days)
        y_margin= torch.tensor(y_margin)
        conditioning = torch.tensor(conditioning)

        print('X shape: ', X.shape)
        print('margin shape: ',y_margin.shape)
        
        return X, active_days, y_margin, conditioning, y_benchmark