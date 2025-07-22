// multiple rider case attempt
/*
Passenger List: elevator maintains list of passengers (pairs of person_id and destination floor)
Batch Processing: elevator picks up as many passengers as possible from the queue if they are on the same floor, respecting the maximum occupancy
Route Management: elevator manages passengers' routes, moving to each passenger's destination floor and logging the necessary information
Complete Logging: detailed logs for each step including entering and exiting the elevator are added to provide comprehensive traceability
also
Occupancy Declaration: occupancy variable initialized to 0 at the start of the elevator function
Occupancy Management: increment occupancy each time a passenger enters the elevator and decrement each time a passenger exits.
*/

#ifndef ELEVATOR_HPP
#define ELEVATOR_HPP

#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <chrono>
#include <random>
#include <atomic>
#include <vector>
#include <condition_variable>
#include <tuple>
#include <unordered_map>

using namespace std;

const int NUM_FLOORS = 50;
const int NUM_ELEVATORS = 6;
const int MAX_OCCUPANCY = 5;
const int MAX_WAIT_TIME = 5000; // Milliseconds

mutex cout_mtx;
mutex queue_mtx;
condition_variable cv; // Didn't end up using this...
queue<tuple<int, int, int>> global_queue; // person_id, start_floor, dest_floor
vector<int> elevator_positions(NUM_ELEVATORS, 0);
atomic<int> num_people_serviced(0);
vector<int> global_passengers_serviced(NUM_ELEVATORS, 0);
int npeople;


void elevator(int id) {
    int occupancy = 0; // Initialize occupancy for each elevator.
    vector<pair<int, int>> passengers; // Pairs of (person_id, dest_floor).
    int current_floor = 0;
    int direction = 1; // (Up = 1, Down = -1)

    // Please complete the code segment...

    // Keep going.
    while(true) {
        // Report info about elevator. 
        {
            lock_guard<mutex> lock(cout_mtx);
            cout << "Elevator " << id << " at Floor " << current_floor << ". Direction: " << (direction > 0 ? "Up" : "Down") << endl;
        }

        // If there are passengers, drop them off.
        if(occupancy != 0) {
            // Keep track of passenegers we want to erase later.
            vector<int> to_erase(passengers.size());
            
            int i = 0;
            int n = passengers.size();
            // Check each person in the elevator.
            while(i < n) {
                // For every person in the elevator, get their id and the floor they want to go.
                pair<int, int> current_pair = passengers[i];
                int current_person = current_pair.first;
                int dest_floor = current_pair.second;

                if(current_floor == dest_floor) {
                    occupancy--;  // If the person is ready to get off, then decrease the occupancy.
                    to_erase[i] = 1; // We will delete the passengers from the passengers vector later.
                    global_passengers_serviced[id]++; // Augment the passengers serviced by this elevator.
                    num_people_serviced++; // Augment total number of passenegers serviced.
                    // Say the persion arrived.
                    {
                        lock_guard<mutex> lock(cout_mtx);
                        cout << "Person " << current_person << " has arived at Floor " << current_floor << "." << endl;
                    }
                } 

                // Augment i.
                i++;
            }

            // Redefine passengers.
            vector<pair<int, int>> temp;
            for(int i = 0; i < to_erase.size(); i++) {
                if(to_erase[i] == 0) {
                    temp.push_back(passengers[i]);
                }
            }
            passengers = move(temp);
        }

        // Pick up people who are waiting.
        {
            // We need to access the global queue.
            lock_guard<mutex> lock(queue_mtx);
            // Make a temporary qlobal queue that we'll need for later.
            queue<tuple<int, int, int>> temp;

            // Look through the global queue. If there is a person that starts on this floor...
            // and the occupancy is less than 10, let them on.
            int n = global_queue.size();
            int i = 0;
        
            // Check if the people in the queue want to get on.
            // While the queue is not empty.
            while(!global_queue.empty()) {
                // Get the first person in line.
                tuple<int, int, int> current_person = global_queue.front();
                int person_id = get<0>(current_person);
                int person_start_floor = get<1>(current_person);
                int person_dest_floor = get<2>(current_person);

                // Take them out of line.
                global_queue.pop();

                // If the person starts on this floor and the elevator isn't full, add them to passengers.
                if(person_start_floor == current_floor) {
                    passengers.emplace_back(person_id, person_dest_floor);
                    occupancy++; // Increase the occupancy.
                    // Print that the person entered the elevator.
                    {
                        lock_guard<mutex> lock(cout_mtx);
                        cout << "Person " << person_id << " entered Elevator " << id << "." << endl;
                    }
                }
                // If not, make sure they go back in the global queue.
                else {
                    temp.emplace(person_id, person_start_floor, person_dest_floor);
                }
            }
            // Make sure the queue is restored.
            global_queue = move(temp);
        }

        // If everyone is done being serviced, just stop right here.
        if(num_people_serviced == npeople && passengers.empty()) break;

        // Move to next floor.
        current_floor += direction;
        
        // If we reach the top, go down. If we reach the bottom, go up.
        if(current_floor == NUM_FLOORS-1 || current_floor == 0) {
            direction *= -1;
        }

        // Update elevator position.
        elevator_positions[id] = current_floor;
        {
            lock_guard<mutex> lock(cout_mtx);
            cout << "Elevator " << id << " moving to Floor " << current_floor << "." << endl;
        }

        // Sleep to simulate moving.
        this_thread::sleep_for(chrono::milliseconds(200));
    }

    // When elevator is done.
    {
            lock_guard<mutex> lock(cout_mtx);
            cout << "Elevator " << id << " has finished servicing all people." << endl;
            cout << "Elevator " << id << " serviced " << global_passengers_serviced[id] << " people." << endl;
    }
}

// Each person exists in a detached thread.
void person(int id) {
    // Randomly select current floor and destination floor for person.
    int curr_floor = rand() % NUM_FLOORS;
    int dest_floor = rand() % NUM_FLOORS;
    
    while (dest_floor == curr_floor) {
        dest_floor = rand() % NUM_FLOORS;
    }

    {
        lock_guard<mutex> lock(cout_mtx);
        cout << "Person " << id << " wants to go from Floor " << curr_floor << " to Floor " << dest_floor << "." << endl;
    }
    
    // Please complete the code segment...

    // The person needs to wait for an elevator to come to their floor.
    // So add this person to the global queue.
    {
        lock_guard<mutex> lock(queue_mtx);
        global_queue.emplace(id, curr_floor, dest_floor);
    }

    // Elevator thread handles everything else.
}

# endif // ELEVATOR_HPP

