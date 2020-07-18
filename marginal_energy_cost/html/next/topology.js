const topologyData = {
    nodes: [
        // ISPs
        { id: 0, x: 400, y: -100, name: "ISP1", device_type: "cloud", color: "grey" },
        { id: 1, x: 600, y: -100, name: "ISP2", device_type: "cloud", color: "grey" },

        // Routers
        { id: 2, x: 400, y: 0, name: "Metropolitan Area 1", device_type: "groupl", color: "red" },
        { id: 3, x: 600, y: 0, name: "Metropolitan Area 2", device_type: "groupl", color: "red" },


        // Switches
        { id: 4, x: 400, y: 100, name: "Data Center Building A", device_type: "groupm" },
        { id: 5, x: 600, y: 100, name: "Data Center Building B", device_type: "groupm" },

        // Servers
        { id: 6, x: 200, y: 200, name: "Server Cluster", device_type: "server" },
        { id: 7, x: 400, y: 200, name: "Server Cluster", device_type: "server" },
        { id: 8, x: 600, y: 200, name: "Server Cluster", device_type: "server" },
        { id: 9, x: 800, y: 200, name: "Server Cluster", device_type: "server" },

        // // SAN
        // { id: 10, x: 500, y: 300, name: "SAN", device_type: "server" }
    ],
    links: [
      // WAN to routers
      { source: 0, target: 2, color: "magenta" },
      { source: 1, target: 3 , color: "magenta" },
  
      // Routers to switches
      { source: 2, target: 4, color: "blue" },
      { source: 2, target: 5, color: "blue" },
      { source: 3, target: 4, color: "blue" },
      { source: 3, target: 5, color: "blue" },
  
      // Switches to Switches
      { source: 4, target: 5, color: "grey" },
      { source: 4, target: 5, color: "grey" },
  
      // Servers to Switches
      { source: 6, target: 4, color: "green" },
      { source: 6, target: 5, color: "red" },
      { source: 7, target: 4, color: "green" },
      { source: 7, target: 5, color: "red" },
      { source: 8, target: 4, color: "green" },
      { source: 8, target: 5, color: "red" },
      { source: 9, target: 4, color: "green" },
      { source: 9, target: 5, color: "red" },
  
      // SAN to Switches
      { source: 10, target: 4, color: "red" },
      { source: 10, target: 4, color: "red" },
      { source: 10, target: 5, color: "red" },
      { source: 10, target: 5, color: "red" }
    ]
};