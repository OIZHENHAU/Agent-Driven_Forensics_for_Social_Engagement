import dotenv from 'dotenv';
import { GoogleGenAI } from "@google/genai";

dotenv.config();

const ai_agent = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });