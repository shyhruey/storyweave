'use client'

import { useState } from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Textarea } from "@/components/ui/textarea"
import { BookOpen, Send, Wand2 } from "lucide-react"

type Message = {
  role: 'user' | 'system'
  content: string
}

export default function StorytellingChatbot() {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'system', content: 'Welcome to the Storytelling Chatbot! Let\'s create a story together.' },
    { role: 'system', content: 'Continue this story: Once upon a time...' }
  ])
  const [input, setInput] = useState('')
  const [summary, setSummary] = useState('')
  const [isTyping, setIsTyping] = useState(false)

  const handleSendMessage = async () => {
    if (input.trim()) {
      const userMessage = { role: 'user', content: input }
      setMessages([...messages, userMessage])
      setInput('')
      setIsTyping(true)

      // Show typing indicator
      setMessages((prevMessages) => [
        ...prevMessages,
        { role: 'system', content: '...' }
      ])

      // Call the API to get the AI's response
      try {
        const response = await fetch('http://localhost:5001/generate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ input })
        })
        
        const data = await response.json()

        // Remove typing indicator and add the first AI response
        setMessages((prevMessages) => [
          ...prevMessages.slice(0, -1), // Remove the last "..." message
          { role: 'system', content: data.response }
        ])

        // Delay before adding the second AI message
        setTimeout(() => {
          setMessages((prevMessages) => [
            ...prevMessages,
            { role: 'system', content: 'What do you think happens next?' } // Second AI message
          ])
        }, 2000) // Delay in milliseconds (2 seconds)
      } catch (error) {
        console.error("Error fetching AI response:", error)
      } finally {
        setIsTyping(false)
      }
    }
  }

  const handleEndStory = () => {
    // Create a story summary starting from "Once upon a time" and filter out "What do you think happens next?"
    const storyContent = messages
      .map(m => m.content)
      .join(' ')
      .replace(/What do you think happens next\?$/, '') // Remove specific ending phrase

    const startIndex = storyContent.indexOf("Once upon a time")
    const filteredSummary = startIndex !== -1 ? storyContent.slice(startIndex) : storyContent
    setSummary(filteredSummary)
  }

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto p-4 bg-background">
      <Card className="flex flex-col h-full">
        <CardHeader>
          <CardTitle className="text-3xl font-bold text-primary flex items-center justify-center">
            <BookOpen className="mr-2" />
            Storytelling Chatbot
          </CardTitle>
        </CardHeader>
        <CardContent className="flex-grow overflow-hidden">
          {/* Setting a max-height and removing flex-grow */}
          <ScrollArea className="h-full max-h-[500px] overflow-y-auto pr-4">
            {messages.map((message, index) => (
              <div key={index} className={`flex mb-4 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`flex items-start max-w-[80%] ${message.role === 'user' ? 'flex-row-reverse' : ''}`}>
                  <Avatar className={`${message.role === 'user' ? 'ml-2' : 'mr-2'}`}>
                    <AvatarFallback>{message.role === 'user' ? 'U' : 'AI'}</AvatarFallback>
                    <AvatarImage src={message.role === 'user' ? '/user-avatar.png' : '/ai-avatar.png'} />
                  </Avatar>
                  <div className={`p-3 rounded-lg ${message.role === 'user' ? 'bg-primary text-primary-foreground' : 'bg-secondary text-secondary-foreground'}`}>
                    {message.content === '...' ? (
                      <div className="typing">
                        <span>.</span>
                        <span>.</span>
                        <span>.</span>
                      </div>
                    ) : (
                      message.content
                    )}
                  </div>
                </div>
              </div>
            ))}

            {/* Display the story summary nested under the last message when available */}
            {summary && (
              <Card className="mt-4 ml-8">
                <CardHeader>
                  <CardTitle className="text-xl font-bold flex items-center">
                    <Wand2 className="mr-2" />
                    Story Summary
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <Textarea readOnly value={summary} className="w-full min-h-[100px]" />
                </CardContent>
              </Card>
            )}
          </ScrollArea>
        </CardContent>
        <CardFooter className="flex flex-col gap-4">
          <div className="flex w-full gap-2">
            <Input
              type="text"
              placeholder="Type your response..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              className="flex-grow"
            />
            <Button onClick={handleSendMessage} size="icon">
              <Send className="h-4 w-4" />
            </Button>
          </div>
          <Button onClick={handleEndStory} className="w-full">
            End Story
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}