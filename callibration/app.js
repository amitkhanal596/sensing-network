/*
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae
All rights reserved.
*/

const express = require('express');
const http = require('http');
const { SerialPort } = require('serialport');
const { ReadlineParser } = require('@serialport/parser-readline');
const net = require('net');
const path = require('path');
const fs = require('fs');

const app = express();
const server = http.createServer(app);
const io = require('socket.io')(server);
const parser = new ReadlineParser({ delimiter: '\r\n' })

let serialPort = null;

// for TCP connection to sensor via WiFi
const simpleSocketClient = new net.Socket();
// for direct connection
const sensorIpAddress = '192.168.4.1';
// for indirect connection (need to change based on the setting)
// const sensorIpAddress = '10.0.0.48';
const sensorPort = 1021;

const findSerialPortPath = SerialPort.list().then(ports => {
  // dynamically find the port path
  let portPath = null;
  ports.forEach(port => {
    if (port.path.includes("/dev/tty.usbmodem" || "/dev/cu.usbmodem")) {
      portPath = port.path;
    }
  });
  return portPath;
});

findSerialPortPath.then(serialPortPath => {
  app.use(express.static('public'))
  app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '/index.html'));
  })

  // The multimodal explorer and the enhanced calibration page, both served
  // alongside the original calibration page, which is left unchanged.
  app.get('/explore', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'explore.html'));
  })

  app.get('/calibrate', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'calibrate.html'));
  })

  if (serialPortPath) {
    console.log("Using serial port connection");
    serialPort = new SerialPort({
      path: serialPortPath,
      baudRate: 9600,
      dataBits: 8,
      parity: 'none',
      stopBits: 1,
      flowControl: false,
      autoOpen: false
    });

    serialPort.open(err => {
      if (err) {
        return console.log('Error opening port: ', err.message)
      }
    })
  } else {
    console.log("Using TCP/IP connection");

    // When there is no serial connection, attempt TCP connection (e.g., via WiFi)
    simpleSocketClient.connect(sensorPort, sensorIpAddress, () => {
      console.log('Connected to the sensor via TCP');
    });

    // Without a handler, a failed connect (no sensor on the network) crashes
    // the server; keep serving the pages so /explore's simulation still works.
    simpleSocketClient.on('error', err => {
      console.log('Sensor not reachable via TCP:', err.message);
    });

    simpleSocketClient.on('close', () => {
      console.log('Connection closed');
    });
  }

  io.on('connection', socket => {
    socket.on('io', data => {
      if (serialPort) {
        parser.on('data', capValue => {
          console.log(capValue); // show data on terminal
          io.emit('data', {
            capValue: capValue,
            connectType: serialPort ? 'serial' : 'tcp/ip'
          }); //send data to socket.js (javascript)
        });
        serialPort.on('open', () => { });
        serialPort.pipe(parser);
        serialPort.write(data.statuses);
      } else {
        simpleSocketClient.on('data', data => {
          const capValue = parseInt(data.toString());
          if (!isNaN(capValue)) {
            console.log(capValue);
            io.emit('data', {
              capValue: capValue,
              connectType: serialPort ? 'serial' : 'tcp/ip'
            }); //send data to socket.js (javascript)
          }
          // simpleSocketClient.destroy(); // kill client after server's response
        });
      }
    });

    const dataDir = path.join(__dirname, 'public', 'data');

    const writeDataFile = (name, contents) => {
      try {
        fs.mkdirSync(dataDir, { recursive: true });
        fs.writeFileSync(path.join(dataDir, name), contents, 'utf8');
        console.log(`wrote ${name}`);
        socket.emit('saved', { file: name });
      } catch (err) {
        console.error(`failed to write ${name}:`, err.message);
        socket.emit('saveError', { file: name, message: err.message });
      }
    };

    socket.on('endCollection', jsonData => writeDataFile('data.json', jsonData));

    socket.on('endProcessing', jsonData => writeDataFile('config.json', jsonData));
  });

  server.listen(8000);
});