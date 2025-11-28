import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:cloud_functions/cloud_functions.dart';

class CloudChatbotService {
  final FirebaseFunctions _functions = FirebaseFunctions.instance;

  // Use Firebase Functions (recommended for Firebase projects)
  Future<Map<String, dynamic>> queryChatbot(String query) async {
    try {
      final HttpsCallable callable = _functions.httpsCallable('simpleChatbot');
      final result = await callable.call(<String, dynamic>{
        'query': query,
      });

      return Map<String, dynamic>.from(result.data);
    } catch (e) {
      print('Firebase Functions error: $e');
      // Fallback to REST API if Firebase Functions fail
      return await _queryViaRestApi(query);
    }
  }

  // Alternative: Direct REST API calls (for non-Firebase deployments)
  Future<Map<String, dynamic>> _queryViaRestApi(String query) async {
    const String baseUrl = 'https://your-region-your-project.cloudfunctions.net';

    try {
      final response = await http.post(
        Uri.parse('$baseUrl/simpleChatbot'),
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode({
          'query': query,
        }),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('API call failed: ${response.statusCode}');
      }
    } catch (e) {
      print('REST API error: $e');
      return {
        'error': 'Service unavailable',
        'answer': 'I apologize, but the chatbot service is currently unavailable. Please try again later.',
        'query': query,
      };
    }
  }

  // Health check
  Future<bool> checkServiceHealth() async {
    try {
      final HttpsCallable callable = _functions.httpsCallable('healthCheck');
      final result = await callable.call();
      return result.data['status'] == 'healthy';
    } catch (e) {
      print('Health check failed: $e');
      return false;
    }
  }

  // Batch queries for efficiency
  Future<List<Map<String, dynamic>>> queryMultiple(List<String> queries) async {
    final results = <Map<String, dynamic>>[];

    for (final query in queries) {
      final result = await queryChatbot(query);
      results.add(result);

      // Small delay to avoid rate limiting
      await Future.delayed(const Duration(milliseconds: 100));
    }

    return results;
  }

  // Cache responses locally for offline use
  final Map<String, Map<String, dynamic>> _responseCache = {};

  Future<Map<String, dynamic>> queryWithCache(String query) async {
    final cacheKey = query.toLowerCase().trim();

    if (_responseCache.containsKey(cacheKey)) {
      return _responseCache[cacheKey]!;
    }

    final result = await queryChatbot(query);

    // Cache successful responses
    if (!result.containsKey('error')) {
      _responseCache[cacheKey] = result;
    }

    return result;
  }

  // Clear cache
  void clearCache() {
    _responseCache.clear();
  }

  // Get cache size
  int get cacheSize => _responseCache.length;
}

// Usage example in Flutter widget:
/*
class ChatbotScreen extends StatefulWidget {
  @override
  _ChatbotScreenState createState() => _ChatbotScreenState();
}

class _ChatbotScreenState extends State<ChatbotScreen> {
  final CloudChatbotService _chatbotService = CloudChatbotService();
  final TextEditingController _controller = TextEditingController();
  List<Map<String, dynamic>> _messages = [];

  Future<void> _sendMessage() async {
    final query = _controller.text.trim();
    if (query.isEmpty) return;

    setState(() {
      _messages.add({'type': 'user', 'text': query});
      _controller.clear();
    });

    try {
      final response = await _chatbotService.queryChatbot(query);

      setState(() {
        _messages.add({
          'type': 'bot',
          'text': response['answer'] ?? 'Sorry, I couldn\'t understand that.',
          'sources': response['sources'] ?? [],
          'confidence': response['confidence'] ?? 0.0,
        });
      });
    } catch (e) {
      setState(() {
        _messages.add({
          'type': 'bot',
          'text': 'Service temporarily unavailable. Please try again.',
        });
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Afya Chatbot')),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                final message = _messages[index];
                return ListTile(
                  title: Text(
                    message['text'],
                    style: TextStyle(
                      color: message['type'] == 'user' ? Colors.blue : Colors.black,
                    ),
                  ),
                );
              },
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _controller,
                    decoration: InputDecoration(
                      hintText: 'Ask about pregnancy and fetal health...',
                      border: OutlineInputBorder(),
                    ),
                    onSubmitted: (_) => _sendMessage(),
                  ),
                ),
                IconButton(
                  icon: Icon(Icons.send),
                  onPressed: _sendMessage,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
*/