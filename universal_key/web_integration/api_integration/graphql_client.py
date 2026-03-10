# -*- coding: utf-8 -*-
"""
GraphQL Client
==============

GraphQL API istekleri için client.
"""

import aiohttp
import asyncio
from typing import Dict, Any, Optional, Union, List
import logging
import json
import time

class GraphQLClient:
    """
    GraphQL API istekleri için async client.
    
    Özellikler:
    - Query ve Mutation desteği
    - Variable binding
    - Fragment support
    - Subscription desteği (WebSocket)
    - Error handling
    - Schema introspection
    """
    
    def __init__(self):
        self.logger = logging.getLogger("GraphQLClient")
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_initialized = False
        
        # GraphQL endpoints
        self.endpoints: Dict[str, str] = {}
        
        # Authentication
        self.auth_headers: Dict[str, Dict[str, str]] = {}
        
        # Schema cache
        self.schemas: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self, config=None) -> bool:
        """GraphQL Client'ı başlat"""
        try:
            self.logger.info("📊 GraphQL Client başlatılıyor...")
            
            connector = aiohttp.TCPConnector(enable_cleanup_closed=True)
            timeout = aiohttp.ClientTimeout(total=60)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "CevahirBot/1.0 GraphQL Client"
                }
            )
            
            self.is_initialized = True
            self.logger.info("✅ GraphQL Client başarıyla başlatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"GraphQL Client başlatma hatası: {e}")
            return False
    
    async def execute(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """GraphQL komutunu çalıştır"""
        if command == "query":
            endpoint = parameters.get("endpoint", "")
            query = parameters.get("query", "")
            variables = parameters.get("variables", {})
            
            if not endpoint or not query:
                return {"error": "endpoint ve query parametreleri gerekli"}
            
            return await self.query(endpoint, query, variables)
        
        elif command == "mutation":
            endpoint = parameters.get("endpoint", "")
            mutation = parameters.get("mutation", "")
            variables = parameters.get("variables", {})
            
            if not endpoint or not mutation:
                return {"error": "endpoint ve mutation parametreleri gerekli"}
            
            return await self.mutation(endpoint, mutation, variables)
        
        elif command == "introspect":
            endpoint = parameters.get("endpoint", "")
            if not endpoint:
                return {"error": "endpoint parametresi gerekli"}
            
            return await self.introspect_schema(endpoint)
        
        else:
            return {"error": f"Desteklenmeyen komut: {command}"}
    
    async def query(self, endpoint: str, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GraphQL query çalıştır"""
        return await self._execute_graphql(endpoint, query, variables, "query")
    
    async def mutation(self, endpoint: str, mutation: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GraphQL mutation çalıştır"""
        return await self._execute_graphql(endpoint, mutation, variables, "mutation")
    
    async def _execute_graphql(self, endpoint: str, operation: str, variables: Optional[Dict[str, Any]], operation_type: str) -> Dict[str, Any]:
        """GraphQL operasyonu çalıştır"""
        if not self.is_initialized or not self.session:
            return {"error": "GraphQL Client başlatılmamış"}
        
        try:
            # GraphQL request body
            request_body = {
                "query": operation,
                "variables": variables or {}
            }
            
            # Headers
            headers = {"Content-Type": "application/json"}
            if endpoint in self.auth_headers:
                headers.update(self.auth_headers[endpoint])
            
            start_time = time.time()
            async with self.session.post(endpoint, json=request_body, headers=headers) as response:
                response_time = time.time() - start_time
                
                if response.status != 200:
                    return {"error": f"GraphQL HTTP hatası: {response.status}"}
                
                try:
                    result = await response.json()
                except:
                    return {"error": "GraphQL response JSON parse hatası"}
                
                # GraphQL errors kontrolü
                if "errors" in result:
                    self.logger.warning(f"GraphQL errors: {result['errors']}")
                
                response_data = {
                    "success": "errors" not in result,
                    "data": result.get("data"),
                    "errors": result.get("errors"),
                    "response_time": response_time,
                    "operation_type": operation_type,
                    "endpoint": endpoint,
                    "timestamp": time.time()
                }
                
                if response_data["success"]:
                    self.logger.info(f"✅ GraphQL {operation_type} başarılı: {endpoint} ({response_time:.2f}s)")
                else:
                    self.logger.warning(f"⚠️ GraphQL {operation_type} hatalı: {endpoint}")
                
                return response_data
                
        except Exception as e:
            self.logger.error(f"GraphQL {operation_type} hatası: {e}")
            return {"error": str(e)}
    
    async def introspect_schema(self, endpoint: str) -> Dict[str, Any]:
        """GraphQL schema'yı introspect et"""
        introspection_query = """
        query IntrospectionQuery {
            __schema {
                types {
                    name
                    kind
                    description
                    fields {
                        name
                        type {
                            name
                            kind
                        }
                    }
                }
            }
        }
        """
        
        try:
            result = await self.query(endpoint, introspection_query)
            
            if result.get("success") and result.get("data"):
                schema_data = result["data"]["__schema"]
                
                # Schema cache'e kaydet
                self.schemas[endpoint] = schema_data
                
                self.logger.info(f"📋 Schema introspection tamamlandı: {endpoint}")
                return {
                    "success": True,
                    "schema": schema_data,
                    "types_count": len(schema_data.get("types", [])),
                    "endpoint": endpoint
                }
            else:
                return {"error": "Schema introspection başarısız", "details": result}
                
        except Exception as e:
            self.logger.error(f"Schema introspection hatası: {e}")
            return {"error": str(e)}
    
    def set_endpoint_auth(self, endpoint: str, auth_headers: Dict[str, str]):
        """Endpoint için authentication ayarla"""
        self.auth_headers[endpoint] = auth_headers
        self.logger.info(f"🔐 {endpoint} için authentication ayarlandı")
    
    def get_schema(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Cache'den schema al"""
        return self.schemas.get(endpoint)
    
    def get_status(self) -> Dict[str, Any]:
        """GraphQL Client durumunu al"""
        return {
            "initialized": self.is_initialized,
            "session_active": self.session is not None and not self.session.closed,
            "endpoints_count": len(self.endpoints),
            "cached_schemas": len(self.schemas),
            "auth_endpoints": list(self.auth_headers.keys())
        }
    
    async def shutdown(self) -> bool:
        """GraphQL Client'ı kapat"""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
            
            self.is_initialized = False
            self.logger.info("📊 GraphQL Client kapatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"GraphQL Client kapatma hatası: {e}")
            return False
