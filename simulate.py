# from: https://github.com/tomasmenito/movie-simpy.git
# backtracking from this article:
# https://realpython.com/simpy-simulating-with-python/

"""
Movies line

- wants to wait at most 10 minutes

"""
import logging
import random
import statistics
from typing import List
from pathlib import Path
from datetime import datetime

from pydantic import BaseModel, Field
from datetime import timedelta
from itertools import groupby, product, chain
from functools import cached_property
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import simpy

logging.basicConfig(level=logging.INFO)


class EmployeeConfig(BaseModel):
    num_cashiers: int = Field(description="Number of cachiers available")
    num_servers: int = Field(description="Number of servers that are available.")
    num_ushers: int = Field(description="Number ofushers available")

    @property
    def total_employees(self):
        return sum((self.num_cashiers, self.num_servers, self.num_ushers))

    @classmethod
    def generate_employee_config(cls, max_employees) -> List['EmployeeConfig']:
        """Generate the parameters for an EmployeeConfig. number of places is a little strange from the docs: For example, product(A, repeat=4) means the same as product(A, A, A, A).  So in this case, build a tuple of three items."""
        num_places = 3
        theaters = [cls(num_cashiers=config[0], num_servers=config[1], num_ushers=config[2]) for config in product(range(1, max_employees + 1), repeat=num_places) if sum(config) <= max_employees]
        theaters.sort(key=lambda e: e.total_employees)
        return theaters

class Theater_Metric(BaseModel):
    wait_times: List[float] = Field(default_factory=list, description="the time consumed between arrival and entering in to the theater.")

    @cached_property
    def avg_wait_time(self):
        return statistics.mean(self.wait_times)




class Theater:
    """Simulate a theater with some limited resources.
    Cashier: Number of people that can provide a ticket
    Server: Number of people that can vend food.
    Usher: Number of people that can check a ticket and admit them to the movie."""
    def __init__(self, employee_config: EmployeeConfig, env = None):
        # Pysim execution environment
        self.employee_config = employee_config
        self.env = env if env else simpy.Environment()
        # the limited number of cashiers that are available in the Theater.
        self.cashier = simpy.Resource(self.env, self.employee_config.num_cashiers)

        # the limited number of servers that are available in the Theater.
        self.server = simpy.Resource(self.env, self.employee_config.num_servers)

        # the limited number of ushers that are available in the Theater.
        self.usher = simpy.Resource(self.env, self.employee_config.num_ushers)
        self.metric = Theater_Metric()

    def __repr__(self):
        return f"Theater({self.employee_config.num_cashiers=}, {self.employee_config.num_servers=}, {self.employee_config.num_ushers=}, {self.total_employees=})"

    @property
    def total_employees(self):
        return self.employee_config.total_employees

    def purchase_ticket(self, moviegoer):
        """Purchasing a ticket from the theater takes some time.  Yield the random duration needed to purchase a ticket."""
        yield self.env.timeout(random.randint(1, 3))

    def check_ticket(self, moviegoer):
        """Checking the ticket takes some time.  The time taken is fixed at 0.05 time units"""
        yield self.env.timeout(3 / 60)

    def sell_food(self, moviegoer):
        """Selling food takes some time  Yield the random duration needed to sell food."""
        yield self.env.timeout(random.randint(1, 5))

    @cached_property
    def avg_wait_time(self):
        return statistics.mean(self.metric.wait_times)


    def go_to_movies(self, moviegoer):
        # Moviegoer arrives at the theater
        arrival_time = self.env.now

        with self.cashier.request() as request:
            yield request
            yield self.env.process(self.purchase_ticket(moviegoer))

        with self.usher.request() as request:
            yield request
            yield self.env.process(self.check_ticket(moviegoer))

        # The customer may or may not want food.  50/50 chance.
        if random.choice([True, False]):
            with self.server.request() as request:
                yield request
                yield self.env.process(self.sell_food(moviegoer))

        # Moviegoer heads into the theater
        self.metric.wait_times.append(self.env.now - arrival_time)

    def incoming_moviegoers_gen(self):
        """Generate incoming moviegoers.  Start with some that are waiting at open, but then a stream at some arrival rate."""
        # Generate 30 movie goers.
        # First three are simultaneously then at a slow trickle?
        for moviegoer in range(3):
            self.env.process(self.go_to_movies(moviegoer))

        #why 27?
        for _ in range(27):
            yield self.env.timeout(0.20)  # Wait a bit before generating new moviegoer

            # Almost done!...
            moviegoer += 1
            self.env.process(self.go_to_movies(moviegoer))


    def run(self):
        self.env.process(self.incoming_moviegoers_gen())
        self.env.run()
        return self


def timestamp():
    #function to print timstamp for filename
    return datetime.now().strftime("%Y%m%d%H%M%S")

def main():
    # Setup
    figure_dir = Path('./plots')
    random.seed(42)

    max_employees = 40
     # create a bunch of theaters
    # Run the simulation and retrieve the average wait time from each run back into the EmployeeConfig.
    theaters = [t.run() for t in (Theater(employee_config=ec) for ec in  EmployeeConfig.generate_employee_config(max_employees))]

    for k, g in groupby(theaters, key=lambda t: t.total_employees):
        # logging.debug(f"group: {k} count: {len(list(g))}")
        best_config = min(g, key=lambda t: t.metric.avg_wait_time)
        logging.info(f"For {k} employees, best config is {best_config}: {best_config.metric.avg_wait_time}\n")

    df = pd.DataFrame(data=[dict(chain(t.employee_config.model_dump().items(),t.metric.model_dump().items())) for t in theaters])
    df = df.explode('wait_times', ignore_index=True)
    df['total_employees'] = df[['num_cashiers','num_ushers', 'num_servers']].sum(axis=1)

    # Plot a line.  x axis = employee count y axis = minimum duration.
    plt.figure(figsize=(10,10))
    # rotate x axis label by 90 degrees
    plt.xticks(rotation=90)
    sns.boxplot(data=df, x='total_employees', y='wait_times', showfliers=False)
    plt.savefig(figure_dir / f"{timestamp()}_boxplot.png")
    plt.show()

if __name__ == "__main__":
    main()
