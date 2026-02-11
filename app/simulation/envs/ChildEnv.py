from app.simulation.envs.Env import Env
from app.domain.Customer import Customer
import gymnasium as gym
import numpy as np

MAX_VISIBLE_CUSTOMERS = 20
INVALID_CUSTOMER_ID = -1

class ChildEnv(Env):
    def __init__(self, mode, instance=None, scenario=None):
        self._visible_customer_ids = []
        super().__init__(mode=mode, instance=instance, scenario=scenario)

    def _get_candidate_customers(self):
        """
        Return waiting customers compatible with the currently selected server.
        """
        candidates = []
        server = self.current_working_server

        for customer in self.customer_waiting.values():
            if server.avg_service_time[customer.task] > 0:
                candidates.append(customer)

        # Deterministic ordering for reproducible behavior
        candidates.sort(key=lambda c: (c.arrival_time, c.id))
        return candidates

    def _get_action_space(self):
        """
        Get the action space.

        Returns:
            action space compatible with gymnasium
        """
        # One action per visible customer slot + one HOLD action.
        return gym.spaces.Discrete(MAX_VISIBLE_CUSTOMERS + 1)
    
    def _get_observation_space(self):
        """
        Get the observation space.

        Returns: 
            observation space compatible with gymnasium
        """
        # Observation contains customer ids for the visible slots.
        return gym.spaces.Box(
            low=INVALID_CUSTOMER_ID,
            high=10_000,
            shape=(MAX_VISIBLE_CUSTOMERS,),
            dtype=np.int32,
        )
    
    def _get_obs(self):
        """
        Convert internal state to observation format.

        Returns: 
            obs(np.array)
        """
        ### Data that can be extracted from the space:
        # Waiting Customers: dict{customer_id: customer}
        #   customer attributes: id: int, arrival_time: float, task_id: int
        # Appointments (all, even passed): dict{customer_id: appointement}
        #   appointment attributes: time: float, customer_id: int, task_id: int, service_time: float
        # Servers: dict{server_id: server}
        #   server attributes: id: int, avg_service_time: dict{task_id: float}
        # Expected end of server activity: dict{server_id: float}, if 0, server available
        # Current selected server id (int)
        # Current simulation time (float)
        waiting_customers, appointments, servers, expected_end, selected_server_id, sim_time = self._get_state()
        del waiting_customers, appointments, servers, expected_end, selected_server_id, sim_time

        candidates = self._get_candidate_customers()
        self._visible_customer_ids = [c.id for c in candidates[:MAX_VISIBLE_CUSTOMERS]]

        obs = self._visible_customer_ids + [INVALID_CUSTOMER_ID] * (
            MAX_VISIBLE_CUSTOMERS - len(self._visible_customer_ids)
        )
        return np.array(obs, dtype=np.int32)
    
    def _get_customer_from_action(self, action) -> Customer:
        """
        Return customer from action.

        Retunrs:
            Customer, or None if invalid action. 
        """   
        # Customers mus be taken from the customers waiting
        # Waiting Customers: dict{customer_id: customer}
        #   customer attributes: id: int, arrival_time: float, task_id: int
        if not isinstance(action, (int, np.integer)):
            return None
        if action < 0 or action >= MAX_VISIBLE_CUSTOMERS:
            return None
        if action >= len(self._visible_customer_ids):
            return None

        customer_id = self._visible_customer_ids[action]
        customer = self.customer_waiting.get(customer_id)
        if customer is None:
            return None

        # Safety check: must still be compatible with the selected server.
        if self.current_working_server.avg_service_time[customer.task] <= 0:
            return None
        return customer

    def _get_invalid_action_reward(self) -> float: 
        """
        Reward chosen for invalid action.

        Returns:
            reward (float) 
        """  
        # ex: return -10
        return -10.0
    
    def _get_valid_reward(self, customer: Customer) -> float:
        """
        Get valid reward.

        Parameters:
            customer (Customer): customer chosen by the action.

        Returns:
            reward (float)
        """
        waiting_time = self.system_time - customer.arrival_time

        # Base incentive to serve a valid customer.
        reward = 2.0

        # Serve long-waiting customers first.
        reward += min(waiting_time / 15.0, 6.0)

        # Reward appointment compliance around target time.
        appointment = self.appointments.get(customer.id)
        if appointment is not None:
            delay = abs(self.system_time - appointment.time)
            reward += max(0.0, 4.0 - delay / 5.0)

        return reward
    
    def action_masks(self):
        """
        Mask not accepted actions.
        """
        mask = [False] * (MAX_VISIBLE_CUSTOMERS + 1)
        valid_slots = min(len(self._visible_customer_ids), MAX_VISIBLE_CUSTOMERS)

        for idx in range(valid_slots):
            mask[idx] = True

        # HOLD action is always available.
        mask[self._get_hold_action_number()] = True
        return mask
    
    def _get_hold_action_number(self):
        """
        Get the action to tell the server to hold and not assign a customer.
        """
        return MAX_VISIBLE_CUSTOMERS
