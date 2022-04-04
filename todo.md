# To Do
* Playback controls
	* Pause/Play
	* FFW/RWD
	* Speed controls
* Object IDs
* Collision animation
* Advanced satellite creation
	* Eccentricity
	* Semi-major axis
	* Inclination
	* etc
* Improve object info overlay
	* Alpha
	* Modularize
	* Kinetic energy
	* Potential energy
	* Total energy
* [Speculative] Temperatures
	* Color is determined by temperature
	* Collisions change temperature
		* Each object has a specific heat, temperature, thermal energy
		* Thermal energy is conserved in collision
		* Kinetic energy lost due to inelastic collision converted to thermal energy
		* Surviving object has new temperature calculated: `(Total thermal energy + Lost kinetic energy)/Specific heat)`
* Integrate Python module for actual planet info? (mass, radius, orbital params, etc.)
	* Can allow for quick creation of Solar System, planetary systems, etc.