import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(SereniApp());
}

class SereniApp extends StatelessWidget {
  const SereniApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Sereni Time',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: SplashScreen(),
    );
  }
}

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  // ignore: library_private_types_in_public_api
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    Future.delayed(Duration(seconds: 3), () {
      Navigator.pushReplacement(
        // ignore: use_build_context_synchronously
        context,
        MaterialPageRoute(builder: (context) => SereniHomePage()),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Image.asset('assets/logo.jpeg', height: 100),
            SizedBox(height: 26),
            Text(
              'Sereni Time',
              style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 8),
            Text(
              'your first step to blooming mind',
              style: TextStyle(fontSize: 16, fontStyle: FontStyle.italic),
            ),
          ],
        ),
      ),
    );
  }
}
class SereniHomePage extends StatefulWidget {
  const SereniHomePage({Key? key}) : super(key: key);

  @override
  _SereniHomePageState createState() => _SereniHomePageState();
}

class _SereniHomePageState extends State<SereniHomePage> {
  final TextEditingController _textController = TextEditingController();
  String _solution = '';
  bool _isLoading = false;
  List<String> _storedInputs = [];
  int _inputCount = 0;
  bool _limitReached = false;

  Future<void> _getSolutionFromApi(String prompt) async {
    setState(() {
      _isLoading = true;
      _solution = ''; // Clear previous solution
    });

    final url = Uri.parse('http://YOUR_IP_ADDRESS:5000/predict'); // Replace with your IP

    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'prompt': prompt}),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _solution = data['solution'];
          _isLoading = false;
        });
      } else {
        setState(() {
          _solution = 'Error: ${response.statusCode}';
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        _solution = 'Error: $e';
        _isLoading = false;
      });
    }
  }

  void _processInput(String input) {
    setState(() {
      if (_inputCount < 7) {
        _storedInputs.add(input);
        _inputCount++;
        _textController.clear();
        _getSolutionFromApi(input); // Call the API here
        print('Stored Input: $_storedInputs, Count: $_inputCount');
      } else {
        _limitReached = true;
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: <Widget>[
          Positioned.fill(
            child: Image.asset(
              'assets/bkgd.jpeg',
              fit: BoxFit.cover,
            ),
          ),
          SafeArea(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.end,
              children: <Widget>[
                if (_limitReached)
                  Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Container(
                      padding: EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.7),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Text(
                        "Dear user, your solving of problems is done for today, and it's enough. Now I have faith in you that you will be able to do your tasks with a calmer mind. Don't worry, I am here with you. Tomorrow again I will take up your problems. Wishing you a peaceful day ahead. ",
                        style: TextStyle(fontSize: 16),
                        textAlign: TextAlign.center,
                      ),
                    ),
                  )
                else
                  Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: TextField(
                      controller: _textController,
                      decoration: InputDecoration(
                        hintText: 'Type your thoughts...',
                        filled: true,
                        fillColor: Colors.white.withOpacity(0.8),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(25.0),
                          borderSide: BorderSide.none,
                        ),
                        suffixIcon: IconButton(
                          icon: Icon(Icons.send),
                          onPressed: () {
                            _processInput(_textController.text);
                          },
                        ),
                      ),
                      onSubmitted: (value) {
                        _processInput(value);
                      },
                    ),
                  ),
                if (_isLoading)
                  CircularProgressIndicator()
                else if (_solution.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: Text(
                      'Solution: $_solution',
                      style: TextStyle(color: Colors.white),
                      textAlign: TextAlign.center,
                    ),
                  ),
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Text(
                    "Sereni time gives calming solutions for temporary period so that you can get out of entangling thoughts and concentrate on your daily things. If you are experiencing severe / persistent anxiety, please contact a medical professional immediately.",
                    style: TextStyle(fontSize: 12, color: Colors.white),
                    textAlign: TextAlign.center,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}